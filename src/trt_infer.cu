#include <chrono>
#include <cublas_v2.h>
#include <unordered_map>
#include "NvInferPlugin.h"
#include "NvUffParser.h"
#include "ros/ros.h"
#include "cv_bridge/cv_bridge.h"
#include "object_msgs/ObjectsInBoxes.h"
#include "ssd_ros/common.h"
#include "ssd_ros/utils.h"

using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;

static constexpr int INPUT_C = 3;
static constexpr int INPUT_H = 300;
static constexpr int INPUT_W = 300;

const char* INPUT_BLOB_NAME = "Input";
const char* OUTPUT_BLOB_NAME0 = "NMS";
int OUTPUT_CLS_SIZE;

DetectionOutputParameters detectionOutputParam{ true, false, 0,   OUTPUT_CLS_SIZE,        100,
                                                100,  0.5,   0.6, CodeTypeSSD::TF_CENTER, { 0, 2, 1 },
                                                true, true };

// Visualization
float visualizeThreshold;

class Logger : public ILogger
{
  void log(Severity severity, const char* msg) override
  {
    if (severity != Severity::kINFO)
      ROS_INFO("[[trt_infer.cu]] %s", msg);
  }
} gLogger;

class FlattenConcat : public IPluginV2
{
public:
    FlattenConcat(int concatAxis, bool ignoreBatch)
        : mIgnoreBatch(ignoreBatch)
        , mConcatAxisID(concatAxis)
    {
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
    }
    //clone constructor
    FlattenConcat(int concatAxis, bool ignoreBatch, int numInputs, int outputConcatAxis, int* inputConcatAxis)
        : mIgnoreBatch(ignoreBatch)
        , mConcatAxisID(concatAxis)
        , mOutputConcatAxis(outputConcatAxis)
        , mNumInputs(numInputs)
    {
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        for (int i = 0; i < mNumInputs; ++i)
            mInputConcatAxis[i] = inputConcatAxis[i];
    }

    FlattenConcat(const void* data, size_t length)
    {
        const char *d = reinterpret_cast<const char*>(data), *a = d;
        mIgnoreBatch = read<bool>(d);
        mConcatAxisID = read<int>(d);
        assert(mConcatAxisID == 1 || mConcatAxisID == 2 || mConcatAxisID == 3);
        mOutputConcatAxis = read<int>(d);
        mNumInputs = read<int>(d);
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        CHECK(cudaMallocHost((void**) &mCopySize, mNumInputs * sizeof(int)));

        std::for_each(mInputConcatAxis, mInputConcatAxis + mNumInputs, [&](int& inp) { inp = read<int>(d); });

        mCHW = read<nvinfer1::DimsCHW>(d);

        std::for_each(mCopySize, mCopySize + mNumInputs, [&](size_t& inp) { inp = read<size_t>(d); });

        assert(d == a + length);
    }
    ~FlattenConcat()
    {
        if (mInputConcatAxis)
            CHECK(cudaFreeHost(mInputConcatAxis));
        if (mCopySize)
            CHECK(cudaFreeHost(mCopySize));
    }
    int getNbOutputs() const override { return 1; }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims >= 1);
        assert(index == 0);
        mNumInputs = nbInputDims;
        CHECK(cudaMallocHost((void**) &mInputConcatAxis, mNumInputs * sizeof(int)));
        mOutputConcatAxis = 0;
#ifdef SSD_INT8_DEBUG
        std::cout << " Concat nbInputs " << nbInputDims << "\n";
        std::cout << " Concat axis " << mConcatAxisID << "\n";
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 3; ++j)
                std::cout << " Concat InputDims[" << i << "]"
                          << "d[" << j << " is " << inputs[i].d[j] << "\n";
#endif
        for (int i = 0; i < nbInputDims; ++i)
        {
            int flattenInput = 0;
            assert(inputs[i].nbDims == 3);
            if (mConcatAxisID != 1)
                assert(inputs[i].d[0] == inputs[0].d[0]);
            if (mConcatAxisID != 2)
                assert(inputs[i].d[1] == inputs[0].d[1]);
            if (mConcatAxisID != 3)
                assert(inputs[i].d[2] == inputs[0].d[2]);
            flattenInput = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
            mInputConcatAxis[i] = flattenInput;
            mOutputConcatAxis += mInputConcatAxis[i];
        }

        return DimsCHW(mConcatAxisID == 1 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 2 ? mOutputConcatAxis : 1,
                       mConcatAxisID == 3 ? mOutputConcatAxis : 1);
    }

    int initialize() override
    {
        CHECK(cublasCreate(&mCublas));
        return 0;
    }

    void terminate() override
    {
        CHECK(cublasDestroy(mCublas));
    }

    size_t getWorkspaceSize(int) const override { return 0; }

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) override
    {
        int numConcats = 1;
        assert(mConcatAxisID != 0);
        numConcats = std::accumulate(mCHW.d, mCHW.d + mConcatAxisID - 1, 1, std::multiplies<int>());

        if (!mIgnoreBatch)
            numConcats *= batchSize;

        float* output = reinterpret_cast<float*>(outputs[0]);
        int offset = 0;
        for (int i = 0; i < mNumInputs; ++i)
        {
            const float* input = reinterpret_cast<const float*>(inputs[i]);
            float* inputTemp;
            CHECK(cudaMalloc(&inputTemp, mCopySize[i] * batchSize));

            CHECK(cudaMemcpyAsync(inputTemp, input, mCopySize[i] * batchSize, cudaMemcpyDeviceToDevice, stream));

            for (int n = 0; n < numConcats; ++n)
            {
                CHECK(cublasScopy(mCublas, mInputConcatAxis[i],
                                  inputTemp + n * mInputConcatAxis[i], 1,
                                  output + (n * mOutputConcatAxis + offset), 1));
            }
            CHECK(cudaFree(inputTemp));
            offset += mInputConcatAxis[i];
        }

        return 0;
    }

    size_t getSerializationSize() const override
    {
        return sizeof(bool) + sizeof(int) * (3 + mNumInputs) + sizeof(nvinfer1::Dims) + (sizeof(mCopySize) * mNumInputs);
    }

    void serialize(void* buffer) const override
    {
        char *d = reinterpret_cast<char*>(buffer), *a = d;
        write(d, mIgnoreBatch);
        write(d, mConcatAxisID);
        write(d, mOutputConcatAxis);
        write(d, mNumInputs);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mInputConcatAxis[i]);
        }
        write(d, mCHW);
        for (int i = 0; i < mNumInputs; ++i)
        {
            write(d, mCopySize[i]);
        }
        assert(d == a + getSerializationSize());
    }

    void configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize) override
    {
        assert(nbOutputs == 1);
        mCHW = inputs[0];
        assert(inputs[0].nbDims == 3);
        CHECK(cudaMallocHost((void**) &mCopySize, nbInputs * sizeof(int)));
        for (int i = 0; i < nbInputs; ++i)
        {
            mCopySize[i] = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2] * sizeof(float);
        }
    }

    bool supportsFormat(DataType type, PluginFormat format) const override
    {
        return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
    }
    const char* getPluginType() const override { return "FlattenConcat_TRT"; }

    const char* getPluginVersion() const override { return "1"; }

    void destroy() override { delete this; }

    IPluginV2* clone() const override
    {
        return new FlattenConcat(mConcatAxisID, mIgnoreBatch, mNumInputs, mOutputConcatAxis, mInputConcatAxis);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    size_t* mCopySize = nullptr;
    bool mIgnoreBatch{false};
    int mConcatAxisID{0}, mOutputConcatAxis{0}, mNumInputs{0};
    int* mInputConcatAxis = nullptr;
    nvinfer1::Dims mCHW;
    cublasHandle_t mCublas;
    std::string mNamespace;
};

namespace
{
const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
const char* FLATTENCONCAT_PLUGIN_NAME{"FlattenConcat_TRT"};
} // namespace

class FlattenConcatPluginCreator : public IPluginCreator
{
public:
    FlattenConcatPluginCreator()
    {
        mPluginAttributes.emplace_back(PluginField("axis", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("ignoreBatch", nullptr, PluginFieldType::kINT32, 1));

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    ~FlattenConcatPluginCreator() {}

    const char* getPluginName() const override { return FLATTENCONCAT_PLUGIN_NAME; }

    const char* getPluginVersion() const override { return FLATTENCONCAT_PLUGIN_VERSION; }

    const PluginFieldCollection* getFieldNames() override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "axis"))
            {
                assert(fields[i].type == PluginFieldType::kINT32);
                mConcatAxisID = *(static_cast<const int*>(fields[i].data));
            }
            if (!strcmp(attrName, "ignoreBatch"))
            {
                assert(fields[i].type == PluginFieldType::kINT32);
                mIgnoreBatch = *(static_cast<const bool*>(fields[i].data));
            }
        }

        return new FlattenConcat(mConcatAxisID, mIgnoreBatch);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override
    {

        //This object will be deleted when the network is destroyed, which will
        //call Concat::destroy()
        return new FlattenConcat(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const override { return mNamespace.c_str(); }

private:
    static PluginFieldCollection mFC;
    bool mIgnoreBatch{false};
    int mConcatAxisID;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace = "";
};

PluginFieldCollection FlattenConcatPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FlattenConcatPluginCreator);

IRuntime* runtime;
ICudaEngine* engine;
IExecutionContext* context;
cudaStream_t stream;

int nbBindings, inputIndex, outputIndex0, outputIndex1;
vector<void*> buffers;
Dims inputDims;

bool is_initialized = false;

vector<string> CLASSES;

void setup(std::string labelFilename, std::string planFilename, int numClasses, float th)
{
  OUTPUT_CLS_SIZE = numClasses;
  visualizeThreshold = th;

  ifstream labelFile(labelFilename.c_str());
  ifstream planFile(planFilename.c_str());

  if (!labelFile.is_open())
  {
    ROS_INFO("Label Not Found!!!");
    is_initialized = false;
  }
  else if (!planFile.is_open())
  {
    ROS_INFO("Plan Not Found!!!");
    is_initialized = false;
  }
  else
  {
    string line;
    while (getline(labelFile, line))
    {
      CLASSES.push_back(line);
    }

    initLibNvInferPlugins(&gLogger, "");

    ROS_INFO("Begin loading plan...");
    stringstream planBuffer;
    planBuffer << planFile.rdbuf();
    string plan = planBuffer.str();

    ROS_INFO("*** deserializing");
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine((void*)plan.data(), plan.size(), nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    CHECK(cudaStreamCreate(&stream));
    ROS_INFO("End loading plan...");

    // Input and output buffer pointers that we pass to the engine - the engine requires exactly
    // IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly 1 input and 2 output.
    nbBindings = engine->getNbBindings();
    vector<pair<int64_t, DataType>> buffersSizes;
    for (int i = 0; i < nbBindings; ++i)
    {
      Dims dims = engine->getBindingDimensions(i);
      DataType dtype = engine->getBindingDataType(i);

      int64_t eltCount = samplesCommon::volume(dims);
      buffersSizes.push_back(make_pair(eltCount, dtype));
    }

    for (int i = 0; i < nbBindings; ++i)
    {
      auto bufferSizesOutput = buffersSizes[i];
      buffers.push_back(samplesCommon::safeCudaMalloc(bufferSizesOutput.first *
                                                       samplesCommon::getElementSize(bufferSizesOutput.second)));
    }

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings().
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex0 = engine->getBindingIndex(OUTPUT_BLOB_NAME0);
    outputIndex1 = outputIndex0 + 1;  // engine.getBindingIndex(OUTPUT_BLOB_NAME1);

    inputDims = engine->getBindingDimensions(inputIndex);
    is_initialized = true;
  }
}

void destroy(void)
{
  if (is_initialized)
  {
    runtime->destroy();
    engine->destroy();
    context->destroy();
    // Release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex0]));
    CHECK(cudaFree(buffers[outputIndex1]));
  }
  is_initialized = false;
}

object_msgs::ObjectsInBoxes infer(const sensor_msgs::ImageConstPtr& color_msg)
{
  object_msgs::ObjectsInBoxes bboxes;

  // preprocessing
  cv::Mat image = cv_bridge::toCvShare(color_msg, "rgb8")->image;
  cv::Size imsize = image.size();
  cv::resize(image, image, cv::Size(INPUT_W, INPUT_H));
  vector<float> inputData(INPUT_C * INPUT_H * INPUT_W);
  cvImageToTensor(image, &inputData[0], inputDims);
  preprocessInception(&inputData[0], inputDims);

  // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
  CHECK(cudaMemcpyAsync(buffers[inputIndex], &inputData[0], INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));

  auto t_start = chrono::high_resolution_clock::now();
  context->execute(1, &buffers[0]);
  auto t_end = chrono::high_resolution_clock::now();
  float total = chrono::duration<float, milli>(t_end - t_start).count();

  // Host memory for outputs.
  vector<float> detectionOut(detectionOutputParam.keepTopK * 7);
  vector<int> keepCount(1);

  CHECK(cudaMemcpyAsync(&detectionOut[0], buffers[outputIndex0], detectionOutputParam.keepTopK * 7 * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(&keepCount[0], buffers[outputIndex1], sizeof(int), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  for (int i = 0; i < keepCount[0]; ++i)
  {
    float* det = &detectionOut[0] + i * 7;
    if (det[2] < visualizeThreshold)
      continue;

    // Output format for each detection is stored in the below order
    // [image_id, label, confidence, xmin, ymin, xmax, ymax]
    assert((int)det[1] < OUTPUT_CLS_SIZE);
    object_msgs::ObjectInBox bbox;
    bbox.object.object_name = CLASSES[(int)det[1]].c_str();
    bbox.object.probability = det[2];
    bbox.roi.x_offset = det[3] * imsize.width;
    bbox.roi.y_offset = det[4] * imsize.height;
    bbox.roi.width = (det[5] - det[3]) * imsize.width;
    bbox.roi.height = (det[6] - det[4]) * imsize.height;
    bbox.roi.do_rectify = false;
    bboxes.objects_vector.push_back(bbox);
    bboxes.inference_time_ms = total;
  }

  bboxes.header = color_msg->header;

  return bboxes;
}
