const workgroup_size = 64;

const shader_code = `
const worldOrigin = vec3(-1.0, -1.0, -1.0);
const particleRadius = 1.0/ 64.0;
const cellSize = vec3(particleRadius * 2.0);
const gridSize = vec3i(64);

@group(0) @binding(0)
var<storage, read> bPos : array<vec4f>;

@group(0) @binding(1)
var<storage, read_write> bGridParticleHash : array<u32>;

@group(0) @binding(2)
var<storage, read_write> bGridParticleIndexInCell : array<u32>;

@group(0) @binding(3)
var<storage, read_write> dCellCount : array<atomic<u32>>;

fn calcGridPos(p: vec3f) -> vec3i
{
    return vec3i((p - worldOrigin)/cellSize); 
}

fn calcGridHash(gridPos: vec3i) -> u32
{    
    let gp = (gridPos + gridSize) % gridSize;
    return u32(gp.x + (gp.y +  gp.z * gridSize.y)* gridSize.x);
}


@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= arrayLength(&bPos)) 
    {
        return;
    }

    let pos = bPos[idx].xyz;
    let gridPos = calcGridPos(pos);
    let hash = calcGridHash(gridPos);

    bGridParticleHash[idx] = hash;
    bGridParticleIndexInCell[idx] = atomicAdd(&dCellCount[hash], 1);
}
`;

function GetPipeline()
{
    if (!("hash_count" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.hash_count];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.hash_count = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }

    return engine_ctx.cache.pipelines.hash_count;
}


export function HashCount(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups = Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_hash_count;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}

