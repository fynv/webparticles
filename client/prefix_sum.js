const workgroup_size = 64;
const workgroup_size_2x = workgroup_size*2;

const shader_code = `
@group(0) @binding(0)
var<uniform> uCount: u32;

@group(0) @binding(1)
var<storage, read> bInput : array<u32>;

@group(0) @binding(2)
var<storage, read_write> bOutput : array<u32>;

@group(0) @binding(3)
var<storage, read_write> bWGCounter : atomic<u32>;

@group(0) @binding(4)
var<storage, read_write> bWGState : array<atomic<u32>>;

var<workgroup> s_workgroup_idx : u32;
var<workgroup> s_inclusive_prefix : u32;
var<workgroup> s_buf : array<u32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(local_invocation_id) LocalInvocationID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    if (threadIdx == 0)
    {
        s_workgroup_idx = atomicAdd(&bWGCounter, 1);
    }
    workgroupBarrier();

    let blockIdx = s_workgroup_idx; 
    
    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<uCount)
    {
        s_buf[threadIdx] = bInput[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<uCount)
    {
        s_buf[threadIdx + ${workgroup_size}] = bInput[i];
    }

    workgroupBarrier();

    var half_size_group = 1u;
    var size_group = 2u;

    while(half_size_group <= ${workgroup_size})
    {
        let gid = threadIdx/half_size_group;
        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
        i = tid + blockIdx*${workgroup_size_2x};
        if (i<uCount)
        {
            s_buf[tid] = s_buf[gid*size_group + half_size_group -1] + s_buf[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    s_inclusive_prefix = 0;
    if (threadIdx == 0)
    {
        atomicStore(&bWGState[blockIdx*3 + 1],  s_buf[${workgroup_size_2x} - 1]);
        atomicStore(&bWGState[blockIdx*3], 1);

        var j = blockIdx;
        while(j>0)
        {
            j--;    
            var state = 0u;
            while(state<1u)
            {
                state = atomicLoad(&bWGState[j*3]);
            }
            
            if (state==2u)
            {
                s_inclusive_prefix+= atomicLoad(&bWGState[j*3 + 2]);
                break;
            }
            else
            {
                s_inclusive_prefix+= atomicLoad(&bWGState[j*3 + 1]);
            }
        }

        atomicStore(&bWGState[blockIdx*3 + 2], s_buf[${workgroup_size_2x} - 1] + s_inclusive_prefix);
        atomicStore(&bWGState[blockIdx*3], 2);
    }
    workgroupBarrier();

    i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<uCount)
    {
        bOutput[i] = s_buf[threadIdx] + s_inclusive_prefix;
    }
    
    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<uCount)
    {
        bOutput[i] = s_buf[threadIdx + ${workgroup_size}] + s_inclusive_prefix;
    }
}
`;


function GetPipeline()
{
    if (!("prefix_sum" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.prefix_sum];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.prefix_sum = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });

    }
    return engine_ctx.cache.pipelines.prefix_sum;
}

export function PrefixSum(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_cells = psystem.numGridCells;    
    let num_groups = Math.floor((num_cells + workgroup_size_2x - 1)/workgroup_size_2x);
    let bind_group = psystem.bind_group_prefix_sum;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();

  
}


