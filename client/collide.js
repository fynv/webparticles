const workgroup_size = 64;

const shader_code = `

const worldOrigin = vec3(-1.0, -1.0, -1.0);
const particleRadius = 1.0/ 64.0;
const cellSize = vec3(particleRadius * 2.0);
const gridSize = vec3i(64);
const spring = 0.5;
const damping = 0.02;
const shear = 0.1;

@group(0) @binding(0)
var<uniform> uNumParicles: u32;

@group(0) @binding(1)
var<storage, read_write> bVel : array<vec4f>;

@group(0) @binding(2)
var<storage, read> bSortedPos : array<vec4f>;

@group(0) @binding(3)
var<storage, read> bSortedVel : array<vec4f>;

@group(0) @binding(4)
var<storage, read> bGridParticleIndex : array<u32>;

@group(0) @binding(5)
var<storage, read> bCellPrefixSum : array<u32>;

fn calcGridPos(p: vec3f) -> vec3i
{
    return vec3i((p - worldOrigin)/cellSize); 
}

fn calcGridHash(gridPos: vec3i) -> u32
{    
    let gp = (gridPos + gridSize) % gridSize;
    return u32(gp.x + (gp.y +  gp.z * gridSize.y)* gridSize.x);
}


fn collideSpheres(posA: vec3f, posB: vec3f, velA: vec3f, velB: vec3f) -> vec3f
{
    let relPos = posB - posA;

    let dist = length(relPos);
    let collideDist = particleRadius + particleRadius;

    var force = vec3(0.0);

    if (dist < collideDist)
    {
        let norm = relPos / dist;
        let relVel = velB - velA;
        let tanVel = relVel - (dot(relVel, norm) * norm);

        force = - spring * (collideDist - dist) * norm;
        force += damping * relVel;
        force += shear * tanVel;
    }
    return force;
}

fn collideCell(gridPos: vec3i, idx: u32, pos: vec3f, vel: vec3f) -> vec3f
{
    let gridHash = calcGridHash(gridPos);
    let startIndex = select(0, bCellPrefixSum[gridHash-1], gridHash>0);
    let endIndex =  bCellPrefixSum[gridHash];
    
    var force = vec3(0.0);
    if (startIndex < endIndex)
    {
        for (var j = startIndex; j< endIndex; j++)
        {
            if (j!= idx)
            {
                let pos2 = bSortedPos[j].xyz;
                let vel2 = bSortedVel[j].xyz;
                force +=  collideSpheres(pos, pos2, vel, vel2);
            }
        }
    }

    return force;

}

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = GlobalInvocationID.x;
    if (idx >= uNumParicles) 
    {
        return;
    }

    let pos = bSortedPos[idx].xyz;
    let vel = bSortedVel[idx].xyz;
    let gridPos = calcGridPos(pos);

    var force = vec3(0.0);

    for (var z=-1; z<=1; z++)
    {
        for (var y=-1; y<=1; y++)
        {
            for (var x=-1; x<=1; x++)
            {
                let neighbourPos = gridPos + vec3(x,y,z);
                force += collideCell(neighbourPos, idx, pos, vel);
            }
        }
    }

    let originalIndex = bGridParticleIndex[idx]; 
    bVel[originalIndex] = vec4(vel + force, 0.0);
}

`;

function GetPipeline()
{
    if (!("collide" in engine_ctx.cache.pipelines))
    {
        let shaderModule = engine_ctx.device.createShaderModule({ code: shader_code });
        let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.collide];
        const pipelineLayoutDesc = { bindGroupLayouts };
        let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

        engine_ctx.cache.pipelines.collide = engine_ctx.device.createComputePipeline({
            layout,
            compute: {
                module: shaderModule,
                entryPoint: 'main',
            },
        });
    }
    return engine_ctx.cache.pipelines.collide;
}

export function Collide(commandEncoder, psystem)
{
    let pipeline = GetPipeline();

    let num_particles= psystem.numParticles;
    let num_groups =  Math.floor((num_particles + workgroup_size - 1)/workgroup_size);
    let bind_group = psystem.bind_group_collide;

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bind_group);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();
}

