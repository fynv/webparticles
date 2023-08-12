const workgroup_size = 64;

const shader_code = `
struct Params
{
    numParticles: i32,
    deltaTime: f32
};

const gravity = vec3(0.0, -0.0003, 0.0);
const globalDamping = 1.0;
const particleRadius = 1.0/ 64.0;
const boundaryDamping = -0.5;

@group(0) @binding(0)
var<uniform> uParams: Params;

@group(0) @binding(1)
var<storage, read_write> bPos : array<vec4f>;

@group(0) @binding(2)
var<storage, read_write> bVel : array<vec4f>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = i32(GlobalInvocationID.x);
    if (idx >=uParams.numParticles) 
    {
        return;
    }

    var pos = bPos[idx].xyz;
    var vel = bVel[idx].xyz;

    vel += gravity * uParams.deltaTime;
    vel *= globalDamping;

    pos += vel * uParams.deltaTime;

    if (pos.x > 1.0 - particleRadius)
    {
        pos.x = 1.0 - particleRadius;
        vel.x *= boundaryDamping;
    }

    if (pos.x < -1.0 + particleRadius)
    {
        pos.x = -1.0 + particleRadius;
        vel.x *= boundaryDamping;
    }

    if (pos.y > 1.0 - particleRadius)
    {
        pos.y = 1.0 - particleRadius;
        vel.y *= boundaryDamping;
    }

    if (pos.y < -1.0 + particleRadius)
    {
        pos.y = -1.0 + particleRadius;
        vel.y *= boundaryDamping;
    }

    if (pos.z > 1.0 - particleRadius)
    {
        pos.z = 1.0 - particleRadius;
        vel.z *= boundaryDamping;
    }

    if (pos.z < -1.0 + particleRadius)
    {
        pos.z = -1.0 + particleRadius;
        vel.z *= boundaryDamping;
    }

    bPos[idx] = vec4(pos, 1.0);
    bVel[idx] = vec4(vel, 0.0);
}
`;

function GetPipeline()
{
    if (!("integrate" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.integrate];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.integrate = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.integrate;
}

export function Integrate(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups =  Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_integrate;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}

