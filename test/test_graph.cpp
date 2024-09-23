#include "catch_amalgamated.hpp"
#include <iostream>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_debug.h>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <random>
#include <unordered_set>
#include <vector>

/// Set any layout according to the connection relationship of partitions
///
/// @param partitions a list of partitions
/// @param id_to_set_any_layout a set of ids of logical tensors with any layout
///     type
void set_any_layout(const std::vector<dnnl::graph::partition>& partitions,
                    std::unordered_set<size_t>& id_to_set_any_layout)
{
    // mapping from output tensor id to the all supported flags of
    // supported partitions, we may only need outputs' supported flags
    std::unordered_map<size_t, std::vector<bool>> output_to_flag_map;
    for(const auto& p: partitions) {
        for(const auto& out: p.get_output_ports()) {
            size_t id = out.get_id();
            if(p.is_supported()
               && output_to_flag_map.find(id)
                      == output_to_flag_map.end()) {
                output_to_flag_map[id] = {};
            }
        }

        for(const auto& in: p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            if(iter != output_to_flag_map.end()) {
                // collect all of supported flags of this tensor's uses
                // Considering we have such a graph:
                //
                //   partition_A  partition_B
                //        \           |
                //      tensor1    tensor2
                //           \     /     |
                //         partition_C  unsupported partition
                //              |
                //           tensor3
                //              |
                //          framework op
                //
                // so the mapping of partition_A's output will be { true }
                // the mapping of partition_B's output will be { true, false }
                // The mapping of partition_C's output will be { false }
                // Only when all supported flags are true, users can set any
                // layout.
                iter->second.push_back(p.is_supported());
            }
        }
    }

    for(const auto& p: partitions) {
        // no need to set `any` layout if this partition is not supported
        if(!p.is_supported())
            continue;
        for(const auto& in: p.get_input_ports()) {
            size_t id = in.get_id();
            auto iter = output_to_flag_map.find(id);
            // if this input tensor is not an output of another supported
            // partition, just skip
            if(iter == output_to_flag_map.end())
                continue;
            std::vector<bool> flag_vec = iter->second;
            // check if all of uses of this tensor are supported partitions,
            // if not, no need to set ANY layout.
            bool need_set_any = std::all_of(flag_vec.begin(), flag_vec.end(),
                                            [](const bool a) {
                                                return a;
                                            });
            if(!need_set_any)
                continue;

            /// record the id of logical tensor that will be set to ANY layout
            id_to_set_any_layout.insert(id);
        }
    }
}

struct cpu_deletor
{
    cpu_deletor() = default;
    void operator()(void* ptr)
    {
        if(ptr)
            free(ptr);
    }
};

void allocate_graph_mem(std::vector<dnnl::graph::tensor>& tensors,
                        const std::vector<dnnl::graph::logical_tensor>& lts,
                        std::vector<std::shared_ptr<void>>& data_buffer,
                        const dnnl::engine& eng)
{
    tensors.reserve(lts.size());
    for(const auto& lt: lts) {
        const auto mem_size = lt.get_mem_size();

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(malloc(mem_size), cpu_deletor{});

        dnnl::graph::tensor new_ts{lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);
    }
}

void allocate_graph_mem(std::vector<dnnl::graph::tensor>& tensors,
                        const std::vector<dnnl::graph::logical_tensor>& lts,
                        std::vector<std::shared_ptr<void>>& data_buffer,
                        std::unordered_map<size_t, dnnl::graph::tensor>& global_outputs_ts_map,
                        const dnnl::engine& eng, bool is_input)
{
    tensors.reserve(lts.size());
    for(const auto& lt: lts) {
        const auto lt_id = lt.get_id();
        const auto mem_size = lt.get_mem_size();

        // check if the input is an output of another partition
        if(is_input) {
            auto pos = global_outputs_ts_map.find(lt_id);
            if(pos != global_outputs_ts_map.end()) {
                tensors.push_back(pos->second);
                continue;
            }
        }

        // memory allocation
        data_buffer.push_back({});
        data_buffer.back().reset(malloc(mem_size), cpu_deletor{});

        dnnl::graph::tensor new_ts{lt, eng, data_buffer.back().get()};
        tensors.push_back(new_ts);

        // record the connection relationship between partitions
        if(!is_input)
            global_outputs_ts_map[lt_id] = tensors.back();
    }
}

/// @page graph_cpu_single_op_partition_cpp
/// @section graph_cpu_single_op_partition_cpp_tutorial cpu_single_op_partition_tutorial() function
///
void cpu_single_op_partition_tutorial() {

    using namespace dnnl::graph;
using data_type = logical_tensor::data_type;
using layout_type = logical_tensor::layout_type;
using dim = logical_tensor::dim;
using dims = logical_tensor::dims;

    dim M = 32, K = 1024, N = 2048;

    dims src0_dims {M, K};
    dims src1_dims {K, N};

    /// @page graph_cpu_single_op_partition_cpp
    /// @subsection graph_cpu_single_op_partition_cpp_get_partition Build Graph and Get Partitions
    ///
    /// In this section, we are trying to create a partition containing the
    /// single op `matmul` without building a graph and getting partition.
    ///

    /// Create the `Matmul` op (#dnnl::graph::op) and attaches attributes
    /// to it, including `transpose_a` and `transpose_b`.
    /// @snippet cpu_single_op_partition.cpp Create matmul
    //[Create matmul]
    logical_tensor matmul_src0_desc {0, data_type::f32};
    logical_tensor matmul_src1_desc {1, data_type::f32};
    logical_tensor matmul_dst_desc {2, data_type::f32};
    op matmul(0, op::kind::MatMul, {matmul_src0_desc, matmul_src1_desc},
            {matmul_dst_desc}, "matmul");
    matmul.set_attr<bool>(op::attr::transpose_a, false);
    matmul.set_attr<bool>(op::attr::transpose_b, false);
    //[Create matmul]

    /// @page graph_cpu_single_op_partition_cpp
    /// @subsection graph_cpu_single_op_partition_cpp_compile Compile and Execute Partition
    ///
    /// In the real case, users like framework should provide device information
    /// at this stage. But in this example, we just use a self-defined device to
    /// simulate the real behavior.
    ///
    /// Create a #dnnl::engine. Also, set a user-defined
    /// #dnnl::graph::allocator to this engine.
    ///
    /// @snippet cpu_single_op_partition.cpp Create engine
    //[Create engine]
    allocator alloc {};
    dnnl::engine eng
            = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
    //[Create engine]

    /// Create a #dnnl::stream on a given engine
    ///
    /// @snippet cpu_single_op_partition.cpp Create stream
    //[Create stream]
    dnnl::stream strm {eng};
    //[Create stream]

    // Memory buffers bound to the partition input/output tensors
    // that helps manage the lifetime of these tensors
    std::vector<std::shared_ptr<void>> data_buffer;

    // Mapping from logical tensor id to the concrete shapes.
    // In practical usage, concrete shapes and layouts are not given
    // until compilation stage, hence need this mapping to mock the step.
    std::unordered_map<size_t, dims> concrete_shapes {
            {0, src0_dims}, {1, src1_dims}};

    // Compile and execute the partitions, including the following steps:
    //
    // 1. Update the input/output logical tensors with concrete shape and layout
    // 2. Compile the partition
    // 3. Update the output logical tensors with queried ones after compilation
    // 4. Allocate memory and bind the data buffer for the partition
    // 5. Execute the partition
    //
    // Although they are not part of the APIs, these steps are essential for
    // the integration of Graph API., hence users need to implement similar
    // logic.

    /// Skip building graph and getting partition, and directly create
    /// the single-op partition
    ///
    /// @snippet cpu_single_op_partition.cpp Create partition
    //[Create partition]
    partition part(matmul, dnnl::engine::kind::cpu);
    //[Create partition]
    if (!part.is_supported()) {
        std::cout << "cpu_single_op_partition: Got unsupported partition, "
                     "users need to handle the operators by themselves."
                  << std::endl;
        return;
    }

    std::vector<logical_tensor> inputs = part.get_input_ports();
    std::vector<logical_tensor> outputs = part.get_output_ports();

    // Update input logical tensors with concrete shape and layout
    for (auto &input : inputs) {
        const auto id = input.get_id();
        // Create logical tensor with strided layout
        input = logical_tensor {id, input.get_data_type(), concrete_shapes[id],
                layout_type::strided};
    }

    // Update output logical tensors with concrete shape and layout
    for (auto &output : outputs) {
        const auto id = output.get_id();
        output = logical_tensor {id, output.get_data_type(),
                DNNL_GRAPH_UNKNOWN_NDIMS,
                // do not require concrete shape as the shape will be inferred
                // based on input shapes during compilation
                layout_type::strided};
    }

    /// Compile the partition to generate compiled partition with the
    /// input and output logical tensors.
    ///
    /// @snippet cpu_single_op_partition.cpp Compile partition
    //[Compile partition]
    compiled_partition cp = part.compile(inputs, outputs, eng);
    //[Compile partition]

    // Update output logical tensors with queried one
    for (auto &output : outputs) {
        const auto id = output.get_id();
        output = cp.query_logical_tensor(id);
    }

    // Allocate memory for the partition, and bind the data buffers with
    // input and output logical tensors
    std::vector<tensor> inputs_ts, outputs_ts;
    allocate_graph_mem(inputs_ts, inputs, data_buffer, eng);
    allocate_graph_mem(outputs_ts, outputs, data_buffer, eng);

    /// Execute the compiled partition on the specified stream.
    ///
    /// @snippet cpu_single_op_partition.cpp Execute compiled partition
    //[Execute compiled partition]
    cp.execute(strm, inputs_ts, outputs_ts);
    //[Execute compiled partition]

    // Wait for all compiled partition's execution finished
    strm.wait();

    /// @page graph_cpu_single_op_partition_cpp
    ///
    std::cout << "Graph:" << std::endl
              << " [matmul_src0] [matmul_src1]" << std::endl
              << "       \\       /" << std::endl
              << "         matmul" << std::endl
              << "            |" << std::endl
              << "        [matmul_dst]" << std::endl
              << "Note:" << std::endl
              << " '[]' represents a logical tensor, which refers to "
                 "inputs/outputs of the graph. "
              << std::endl;
}

void cpu_getting_started_tutorial()
{

    using namespace dnnl::graph;
    using data_type = logical_tensor::data_type;
    using layout_type = logical_tensor::layout_type;
    using dim = logical_tensor::dim;
    using dims = logical_tensor::dims;

    dim N = 8, IC = 3, OC1 = 96, OC2 = 96;
    dim IH = 225, IW = 225, KH1 = 11, KW1 = 11, KH2 = 1, KW2 = 1;

    dims conv0_input_dims{N, IC, IH, IW};
    dims conv0_weight_dims{OC1, IC, KH1, KW1};
    dims conv0_bias_dims{OC1};
    dims conv1_weight_dims{OC1, OC2, KH2, KW2};
    dims conv1_bias_dims{OC2};

    /// @page graph_cpu_getting_started_cpp
    /// @subsection graph_cpu_getting_started_cpp_get_partition Build Graph and Get Partitions
    ///
    /// In this section, we are trying to build a graph containing the pattern
    /// `conv0->relu0->conv1->relu1`. After that, we can get all of
    /// partitions which are determined by backend.
    ///

    /// To build a graph, the connection relationship of different ops must be
    /// known. In oneDNN Graph, #dnnl::graph::logical_tensor is used to express
    /// such relationship. So, next step is to create logical tensors for these
    /// ops including inputs and outputs.
    ///
    /// @note It's not necessary to provide concrete shape/layout information
    /// at graph partitioning stage. Users can provide these information till
    /// compilation stage.
    ///
    /// Create input/output #dnnl::graph::logical_tensor for first `Convolution` op.
    /// @snippet cpu_getting_started.cpp Create conv's logical tensor
    //[Create conv's logical tensor]
    logical_tensor conv0_src_desc{0, data_type::f32};
    logical_tensor conv0_weight_desc{1, data_type::f32};
    logical_tensor conv0_dst_desc{2, data_type::f32};
    //[Create conv's logical tensor]

    /// Create first `Convolution` op (#dnnl::graph::op) and attaches attributes
    /// to it, such as `strides`, `pads_begin`, `pads_end`, `data_format`, etc.
    /// @snippet cpu_getting_started.cpp Create first conv
    //[Create first conv]
    op conv0(0, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
             {conv0_dst_desc}, "conv0");
    conv0.set_attr<dims>(op::attr::strides, {4, 4});
    conv0.set_attr<dims>(op::attr::pads_begin, {0, 0});
    conv0.set_attr<dims>(op::attr::pads_end, {0, 0});
    conv0.set_attr<dims>(op::attr::dilations, {1, 1});
    conv0.set_attr<int64_t>(op::attr::groups, 1);
    conv0.set_attr<std::string>(op::attr::data_format, "NCX");
    conv0.set_attr<std::string>(op::attr::weights_format, "OIX");
    //[Create first conv]

    /// Create input/output logical tensors for first `BiasAdd` op and create the first `BiasAdd` op
    /// @snippet cpu_getting_started.cpp Create first bias_add
    //[Create first bias_add]
    logical_tensor conv0_bias_desc{3, data_type::f32};
    logical_tensor conv0_bias_add_dst_desc{
        4, data_type::f32, layout_type::undef};
    op conv0_bias_add(1, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc},
                      {conv0_bias_add_dst_desc}, "conv0_bias_add");
    conv0_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");
    //[Create first bias_add]

    /// Create output logical tensors for first `Relu` op and create the op.
    /// @snippet cpu_getting_started.cpp Create first relu
    //[Create first relu]
    logical_tensor relu0_dst_desc{5, data_type::f32};
    op relu0(2, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc},
             "relu0");
    //[Create first relu]

    /// Create input/output logical tensors for second `Convolution` op and create the second `Convolution` op.
    /// @snippet cpu_getting_started.cpp Create second conv
    //[Create second conv]
    logical_tensor conv1_weight_desc{6, data_type::f32};
    logical_tensor conv1_dst_desc{7, data_type::f32};
    op conv1(3, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc},
             {conv1_dst_desc}, "conv1");
    conv1.set_attr<dims>(op::attr::strides, {1, 1});
    conv1.set_attr<dims>(op::attr::pads_begin, {0, 0});
    conv1.set_attr<dims>(op::attr::pads_end, {0, 0});
    conv1.set_attr<dims>(op::attr::dilations, {1, 1});
    conv1.set_attr<int64_t>(op::attr::groups, 1);
    conv1.set_attr<std::string>(op::attr::data_format, "NCX");
    conv1.set_attr<std::string>(op::attr::weights_format, "OIX");
    //[Create second conv]

    /// Create input/output logical tensors for second `BiasAdd` op and create the op.
    /// @snippet cpu_getting_started.cpp Create second bias_add
    //[Create second bias_add]
    logical_tensor conv1_bias_desc{8, data_type::f32};
    logical_tensor conv1_bias_add_dst_desc{9, data_type::f32};
    op conv1_bias_add(4, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc},
                      {conv1_bias_add_dst_desc}, "conv1_bias_add");
    conv1_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");
    //[Create second bias_add]

    /// Create output logical tensors for second `Relu` op and create the op
    /// @snippet cpu_getting_started.cpp Create second relu
    //[Create second relu]
    logical_tensor relu1_dst_desc{10, data_type::f32};
    op relu1(5, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc},
             "relu1");
    //[Create second relu]

    /// Finally, those created ops will be added into the graph. The graph
    /// inside will maintain a list to store all these ops. To create a graph,
    /// #dnnl::engine::kind is needed because the returned partitions
    /// maybe vary on different devices. For this example, we use CPU engine.
    ///
    /// @note The order of adding op doesn't matter. The connection will
    /// be obtained through logical tensors.
    ///
    /// Create graph and add ops to the graph
    /// @snippet cpu_getting_started.cpp Create graph and add ops
    //[Create graph and add ops]
    graph g(dnnl::engine::kind::cpu);

    g.add_op(conv0);
    g.add_op(conv0_bias_add);
    g.add_op(relu0);

    g.add_op(conv1);
    g.add_op(conv1_bias_add);
    g.add_op(relu1);
    //[Create graph and add ops]

    /// After adding all ops into the graph, call
    /// #dnnl::graph::graph::get_partitions() to indicate that the
    /// graph building is over and is ready for partitioning. Adding new
    /// ops into a finalized graph or partitioning a unfinalized graph
    /// will both lead to a failure.
    ///
    /// @snippet cpu_getting_started.cpp Finalize graph
    //[Finalize graph]
    g.finalize();
    //[Finalize graph]

    /// After finished above operations, we can get partitions by calling
    /// #dnnl::graph::graph::get_partitions().
    ///
    /// In this example, the graph will be partitioned into two partitions:
    /// 1. conv0 + conv0_bias_add + relu0
    /// 2. conv1 + conv1_bias_add + relu1
    ///
    /// @snippet cpu_getting_started.cpp Get partition
    //[Get partition]
    auto partitions = g.get_partitions();
    //[Get partition]

    // Check partitioning results to ensure the examples works. Users do
    // not need to follow this step.
    assert(partitions.size() == 2);

    /// @page graph_cpu_getting_started_cpp
    /// @subsection graph_cpu_getting_started_cpp_compile Compile and Execute Partition
    ///
    /// In the real case, users like framework should provide device information
    /// at this stage. But in this example, we just use a self-defined device to
    /// simulate the real behavior.
    ///
    /// Create a #dnnl::engine. Also, set a user-defined
    /// #dnnl::graph::allocator to this engine.
    ///
    /// @snippet cpu_getting_started.cpp Create engine
    //[Create engine]
    allocator alloc{};
    dnnl::engine eng = make_engine_with_allocator(dnnl::engine::kind::cpu, 0, alloc);
    //[Create engine]

    /// Create a #dnnl::stream on a given engine
    ///
    /// @snippet cpu_getting_started.cpp Create stream
    //[Create stream]
    dnnl::stream strm{eng};
    //[Create stream]

    // Mapping from logical tensor id to output tensors
    // used to the connection relationship between partitions (e.g partition 0's
    // output tensor is fed into partition 1)
    std::unordered_map<size_t, tensor> global_outputs_ts_map;

    // Memory buffers bound to the partition input/output tensors
    // that helps manage the lifetime of these tensors
    std::vector<std::shared_ptr<void>> data_buffer;

    // Mapping from id to queried logical tensor from compiled partition
    // used to record the logical tensors that are previously enabled with
    // ANY layout
    std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;

    // This is a helper function which helps decide which logical tensor is
    // needed to be set with `dnnl::graph::logical_tensor::layout_type::any`
    // layout.
    // This function is not a part to Graph API, but similar logic is
    // essential for Graph API integration to achieve best performance.
    // Typically, users need implement the similar logic in their code.
    std::unordered_set<size_t> ids_with_any_layout;
    set_any_layout(partitions, ids_with_any_layout);

    // Mapping from logical tensor id to the concrete shapes.
    // In practical usage, concrete shapes and layouts are not given
    // until compilation stage, hence need this mapping to mock the step.
    std::unordered_map<size_t, dims> concrete_shapes{{0, conv0_input_dims},
                                                     {1, conv0_weight_dims},
                                                     {3, conv0_bias_dims},
                                                     {6, conv1_weight_dims},
                                                     {8, conv1_bias_dims}};

    // Compile and execute the partitions, including the following steps:
    //
    // 1. Update the input/output logical tensors with concrete shape and layout
    // 2. Compile the partition
    // 3. Update the output logical tensors with queried ones after compilation
    // 4. Allocate memory and bind the data buffer for the partition
    // 5. Execute the partition
    //
    // Although they are not part of the APIs, these steps are essential for
    // the integration of Graph API., hence users need to implement similar
    // logic.
    for(const auto& partition: partitions) {
        if(!partition.is_supported()) {
            std::cout
                << "cpu_get_started: Got unsupported partition, users need "
                   "handle the operators by themselves."
                << std::endl;
            continue;
        }

        std::vector<logical_tensor> inputs = partition.get_input_ports();
        std::vector<logical_tensor> outputs = partition.get_output_ports();

        // Update input logical tensors with concrete shape and layout
        for(auto& input: inputs) {
            const auto id = input.get_id();
            // If the tensor is an output of another partition,
            // use the cached logical tensor
            if(id_to_queried_logical_tensors.find(id)
               != id_to_queried_logical_tensors.end())
                input = id_to_queried_logical_tensors[id];
            else
                // Create logical tensor with strided layout
                input = logical_tensor{id, input.get_data_type(),
                                       concrete_shapes[id], layout_type::strided};
        }

        // Update output logical tensors with concrete shape and layout
        for(auto& output: outputs) {
            const auto id = output.get_id();
            output = logical_tensor{id, output.get_data_type(),
                                    DNNL_GRAPH_UNKNOWN_NDIMS, // set output dims to unknown
                                    ids_with_any_layout.count(id) ? layout_type::any
                                                                  : layout_type::strided};
        }

        /// Compile the partition to generate compiled partition with the
        /// input and output logical tensors.
        ///
        /// @snippet cpu_getting_started.cpp Compile partition
        //[Compile partition]
        compiled_partition cp = partition.compile(inputs, outputs, eng);
        //[Compile partition]

        // Update output logical tensors with queried one
        for(auto& output: outputs) {
            const auto id = output.get_id();
            output = cp.query_logical_tensor(id);
            id_to_queried_logical_tensors[id] = output;
        }

        // Allocate memory for the partition, and bind the data buffers with
        // input and output logical tensors
        std::vector<tensor> inputs_ts, outputs_ts;
        allocate_graph_mem(inputs_ts, inputs, data_buffer,
                           global_outputs_ts_map, eng, /*is partition input=*/true);
        allocate_graph_mem(outputs_ts, outputs, data_buffer,
                           global_outputs_ts_map, eng, /*is partition input=*/false);

        /// Execute the compiled partition on the specified stream.
        ///
        /// @snippet cpu_getting_started.cpp Execute compiled partition
        //[Execute compiled partition]
        cp.execute(strm, inputs_ts, outputs_ts);
        //[Execute compiled partition]
    }

    // Wait for all compiled partition's execution finished
    strm.wait();

    /// @page graph_cpu_getting_started_cpp
    ///
    std::cout << "Graph:" << std::endl
              << " [conv0_src] [conv0_wei]" << std::endl
              << "       \\      /" << std::endl
              << "         conv0" << std::endl
              << "          \\    [conv0_bias_src1]" << std::endl
              << "           \\      /" << std::endl
              << "         conv0_bias_add" << std::endl
              << "                |" << std::endl
              << "              relu0" << std::endl
              << "                \\   [conv1_wei]" << std::endl
              << "                 \\    /" << std::endl
              << "                  conv1" << std::endl
              << "                    \\  [conv1_bias_src1]" << std::endl
              << "                     \\      /" << std::endl
              << "                  conv1_bias_add" << std::endl
              << "                          |" << std::endl
              << "                        relu1" << std::endl
              << "                          |" << std::endl
              << "                      [relu_dst]" << std::endl
              << "Note:" << std::endl
              << " '[]' represents a logical tensor, which refers to "
                 "inputs/outputs of the graph. "
              << std::endl;
}

// Exception class to indicate that the example uses a feature that is not
// available on the current systems. It is not treated as an error then, but
// just notifies a user.
struct example_allows_unimplemented: public std::exception
{
    example_allows_unimplemented(const char* message) noexcept
        : message(message)
    {
    }
    const char* what() const noexcept override
    {
        return message;
    }
    const char* message;
};

inline const char* engine_kind2str_upper(dnnl::engine::kind kind)
{
    if(kind == dnnl::engine::kind::cpu)
        return "CPU";
    if(kind == dnnl::engine::kind::gpu)
        return "GPU";
    assert(!"not expected");
    return "<Unknown engine>";
}

// TBB runtime may crash when it is used under CTest. This is a known TBB
// limitation that can be worked around by doing explicit finalization.
// The API to do that was introduced in 2021.6.0. When using an older TBB
// runtime the crash may still happen.
inline void finalize()
{
#ifdef DNNL_TBB_NEED_EXPLICIT_FINALIZE
    tbb::task_scheduler_handle handle = tbb::task_scheduler_handle{tbb::attach{}};
    tbb::finalize(handle, std::nothrow);
#endif
}

// Runs example function with signature void() and catches errors.
// Returns `0` on success, `1` or oneDNN error, and `2` on example error.
inline int handle_example_errors(
    std::initializer_list<dnnl::engine::kind> engine_kinds,
    std::function<void()> example)
{
    int exit_code = 0;

    try {
        example();
    } catch(example_allows_unimplemented& e) {
        std::cout << e.message << std::endl;
        exit_code = 0;
    } catch(dnnl::error& e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch(std::exception& e) {
        std::cout << "Error in the example: " << e.what() << "." << std::endl;
        exit_code = 2;
    }

    std::string engine_kind_str;
    for(auto it = engine_kinds.begin(); it != engine_kinds.end(); ++it) {
        if(it != engine_kinds.begin())
            engine_kind_str += "/";
        engine_kind_str += engine_kind2str_upper(*it);
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << engine_kind_str << "." << std::endl;
    finalize();
    return exit_code;
}

TEST_CASE("Test Graph"
          "[oneDNN]")
{
    handle_example_errors({dnnl::engine::kind::cpu}, cpu_single_op_partition_tutorial);
}
