#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <algorithm>
#include <vector>
#include <random>

namespace py = pybind11;

using ArrayDict = std::map<std::string, py::array>;

using TensorDict = std::map<std::string, torch::Tensor>;

class NumpyDictSampler
{
public:
    struct ArrayInfo
    {
        float *data_ptr;
        size_t rows;
        size_t cols;
    };

    NumpyDictSampler(py::object data_dict, int batch_size, std::vector<int32_t> indices, int seqlen)
        : batch_size(batch_size), indices(std::move(indices)), seqlen(seqlen)
    {
        // 验证字典并提取数组信息
        py::detail::make_caster<ArrayDict> caster;
        caster.load(data_dict, false);

        ArrayDict dict = std::move(*caster);

        size_t ref_rows = 0;

        for (const auto &item : dict)
        {
            const std::string &key = item.first;
            const py::array &arr = item.second;
            auto buf = arr.request();
            if (buf.ndim != 2)
                throw std::runtime_error("只支持二维数组");

            if (arrays.empty())
                ref_rows = buf.shape[0];
            else if (buf.shape[0] != ref_rows)
                throw std::runtime_error("所有数组行数必须一致");

            arrays[key] = {
                static_cast<float *>(buf.ptr),
                buf.shape[0],
                buf.shape[1]};
        }

        // 初始化索引
        // indices.resize(ref_rows);
        // std::iota(indices.begin(), indices.end(), 0);
    }

    class Iterator
    {
    public:
        Iterator(std::map<std::string, ArrayInfo> arrays,
                 std::vector<int> indices, int batch_size, int seqlen)
            : arrays(std::move(arrays)),
              indices(std::move(indices)),
              batch_size(batch_size),
              seqlen(seqlen),
              rd(),
              gen(rd())
        {
            std::shuffle(this->indices.begin(), this->indices.end(), gen);
            // auto new_size = size_t(this->indices.size() * ratio);
            // this->indices.resize(new_size);
            current = 0;
        }

        Iterator(Iterator &&other) noexcept
            : arrays(std::move(other.arrays)),
              indices(std::move(other.indices)),
              batch_size(other.batch_size),
              seqlen(other.seqlen),
              current(other.current),
              gen(std::move(other.gen))
        {

            other.current = 0;
        }

        TensorDict next()
        {
            if (current >= indices.size())
                throw py::stop_iteration();

            // 生成当前批次索引
            auto start = indices.begin() + current;
            auto end = (current + batch_size <= indices.size()) ? start + batch_size : indices.end();
            current += batch_size;

            // 创建结果字典
            TensorDict batch_dict;
            for (auto &pair : arrays)
            {
                bool add_index = pair.first == arrays.begin()->first;
                const auto &info = pair.second;
                int32_t *indices_data = nullptr;
                if (add_index)
                {
                    auto options = torch::TensorOptions()
                                       .dtype(torch::kInt32)
                                       .device(torch::kCPU)
                                       .pinned_memory(true);

                    batch_dict["indices"] = torch::empty(
                        {static_cast<int64_t>(end - start) * seqlen},
                        options);
                    indices_data = batch_dict["indices"].data_ptr<int32_t>();
                }

                // 创建pin-memory tensor
                auto options = torch::TensorOptions()
                                   .dtype(torch::kFloat32)
                                   .device(torch::kCPU)
                                   .pinned_memory(true);

                std::vector<int64_t> shape = {end - start, seqlen * info.cols};
                torch::Tensor tensor = torch::empty(shape, options);

                // 填充数据
                float *tensor_data = tensor.data_ptr<float>();
                for (auto it = start; it != end; ++it)
                {
                    const float *src = info.data_ptr + (*it) * info.cols;
                    std::memcpy(tensor_data, src, seqlen * info.cols * sizeof(float));
                    tensor_data += info.cols;
                    if (indices_data)
                    {
                        for (size_t s_i = 0; s_i < seqlen; ++s_i)
                        {
                            *(indices_data++) = *(it) + s_i;
                        }
                    }
                }
                tensor_data = tensor.data_ptr<float>();
                batch_dict[pair.first.c_str()] = tensor;
            }
            return batch_dict;
        }

    private:
        std::map<std::string, ArrayInfo> arrays;
        std::vector<int> indices;
        int batch_size;
        int seqlen;
        size_t current;
        std::random_device rd;
        std::mt19937 gen;
    };

    Iterator __iter__()
    {
        return Iterator(arrays, indices, batch_size, seqlen);
    }

private:
    std::map<std::string, ArrayInfo> arrays;
    std::vector<int> indices;
    int batch_size;
    float ratio;
    int seqlen;
};
