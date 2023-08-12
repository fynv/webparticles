import { Integrate } from "./integrate.js"
import { HashCount } from "./hash_count.js"
import { PrefixSum } from "./prefix_sum.js"
import { Scatter } from "./scatter.js"
import { Collide } from "./collide.js"

const workgroup_size = 64;
const workgroup_size_2x = workgroup_size*2;

const ncolors = 7;
const c = [
    [ 1.0, 0.0, 0.0, ],
    [ 1.0, 0.5, 0.0, ],
    [ 1.0, 1.0, 0.0, ],
    [ 0.0, 1.0, 0.0, ],
    [ 0.0, 1.0, 1.0, ],
    [ 0.0, 0.0, 1.0, ],
    [ 1.0, 0.0, 1.0, ]
];

const particleRadius = 1.0 / 64.0;

export class ParticleSystem
{
    constructor(numParticles, gridSize)
    {
        this.numParticles = numParticles;
        this.gridSize = gridSize;

        this.numGridCells = gridSize[0] * gridSize[1] * gridSize[2];
        
        this.gridSortBits = 18;  // increase this for larger grids


        this._initialize();
    }

    
    _initialize()
    {

        this.hPos = new Float32Array(this.numParticles * 4);
        this.hVel = new Float32Array(this.numParticles * 4);

        this.hCellStart = new Int32Array(this.numGridCells);
        this.hCellEnd = new Int32Array(this.numGridCells);

        let memSize = this.numParticles * 4 * 4;
        this.dPos = engine_ctx.createBuffer0(memSize, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.dVel = engine_ctx.createBuffer0(memSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.dSortedPos = engine_ctx.createBuffer0(memSize, GPUBufferUsage.STORAGE);
        this.dSortedVel = engine_ctx.createBuffer0(memSize, GPUBufferUsage.STORAGE);

        this.dGridParticleHash  = engine_ctx.createBuffer0(this.numParticles * 4, GPUBufferUsage.STORAGE);
        this.dGridParticleIndexInCell = engine_ctx.createBuffer0(this.numParticles * 4, GPUBufferUsage.STORAGE);
        this.dGridParticleIndex  = engine_ctx.createBuffer0(this.numParticles * 4, GPUBufferUsage.STORAGE);

        this.dCellCount =  engine_ctx.createBuffer0(this.numGridCells * 4, GPUBufferUsage.STORAGE| GPUBufferUsage.COPY_DST);
        this.dCellPrefixSum =  engine_ctx.createBuffer0(this.numGridCells * 4, GPUBufferUsage.STORAGE);

        let num_groups = Math.floor((this.numGridCells + workgroup_size_2x - 1)/workgroup_size_2x);
        this.buf_workgroup_counter = engine_ctx.createBuffer0(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.buf_workgroup_state = engine_ctx.createBuffer0(num_groups * 3 *4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        
        let hColor = new Float32Array(this.numParticles * 4);
        for (let i=0; i<this.numParticles; i++)
        {
            let t = i/this.numParticles;
            t *= (ncolors - 1);
            let j = Math.floor(t);
            let u =  t - j;
            hColor[i*4] = (1.0 - u) * c[j][0] + u * c[j+1][0];
            hColor[i*4 + 1] = (1.0 - u) * c[j][1] + u * c[j+1][1];
            hColor[i*4 + 2] = (1.0 - u) * c[j][2] + u * c[j+1][2];
            hColor[i*4 + 3] = 1.0;
        }
        
        this.dColor = engine_ctx.createBuffer(hColor.buffer, GPUBufferUsage.VERTEX);

        this.dConstant = engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this.dConstant2 = engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

        let layout_entries_integrate = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "uniform"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
        ];

        let bindGroupLayoutIntegrate = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_integrate });
        engine_ctx.cache.bindGroupLayouts.integrate = bindGroupLayoutIntegrate;

        let layout_entries_hash_count = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "uniform"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },        
        ];

        let bindGroupLayoutHashCount = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_hash_count });
        engine_ctx.cache.bindGroupLayouts.hash_count = bindGroupLayoutHashCount;

        let layout_entries_prefix_sum = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "uniform"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },

        ];

        let bindGroupLayoutPrefixSum = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_prefix_sum });
        engine_ctx.cache.bindGroupLayouts.prefix_sum = bindGroupLayoutPrefixSum;

        let layout_entries_scatter = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "uniform"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 6,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 7,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 8,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },

        ];

        let bindGroupLayoutScatter = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_scatter });
        engine_ctx.cache.bindGroupLayouts.scatter = bindGroupLayoutScatter;

        let layout_entries_collide = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "uniform"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            },

        ];

        let bindGroupLayoutCollide = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_collide });
        engine_ctx.cache.bindGroupLayouts.collide = bindGroupLayoutCollide;

        let group_entries_integrate = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstant
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dPos            
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dVel            
                }
            },
        ];

        this.bind_group_integrate = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutIntegrate, entries: group_entries_integrate});

        let group_entries_hash_count = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstant
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dPos            
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dGridParticleHash
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dGridParticleIndexInCell
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dCellCount
                }
            },

        ];

        this.bind_group_hash_count = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutHashCount, entries: group_entries_hash_count});

        let group_entries_prefix_sum = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstant2
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dCellCount            
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dCellPrefixSum            
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.buf_workgroup_counter            
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.buf_workgroup_state            
                }
            },
        ];

        this.bind_group_prefix_sum = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutPrefixSum, entries: group_entries_prefix_sum});

        let group_entries_scatter = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstant
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dPos            
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dVel            
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dCellPrefixSum            
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dGridParticleHash            
                }
            },
            {
                binding: 5,
                resource:{
                    buffer: this.dGridParticleIndexInCell            
                }
            },
            {
                binding: 6,
                resource:{
                    buffer: this.dSortedPos            
                }
            },
            {
                binding: 7,
                resource:{
                    buffer: this.dSortedVel            
                }
            },
            {
                binding: 8,
                resource:{
                    buffer: this.dGridParticleIndex            
                }
            },
        ];

        this.bind_group_scatter = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutScatter, entries: group_entries_scatter});

        let group_entries_collide = [
            {
                binding: 0,
                resource:{
                    buffer: this.dConstant
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dVel            
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dSortedPos            
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dSortedVel            
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dGridParticleIndex            
                }
            },
            {
                binding: 5,
                resource:{
                    buffer: this.dCellPrefixSum            
                }
            }
        ];

        this.bind_group_collide = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutCollide, entries: group_entries_collide});

    }

    initGrid(size, spacing, jitter)
    {
        for (let z=0; z<size[2]; z++)
        {
            for (let y=0; y<size[1]; y++)
            {
                for (let x=0; x< size[0]; x++)
                {
                    let i = (z*size[1]*size[0]) + (y*size[0]) + x;

                    if (i<this.numParticles)
                    {
                        this.hPos[i*4] = spacing * x + particleRadius - 1.0 + (Math.random() * 2.0 - 1.0) * jitter;
                        this.hPos[i*4 + 1] = spacing * y + particleRadius - 1.0 + (Math.random() * 2.0 - 1.0) * jitter;
                        this.hPos[i*4 + 2] = spacing * z + particleRadius - 1.0 + (Math.random() * 2.0 - 1.0) * jitter;
                        this.hPos[i*4 + 3] = 1.0;

                        this.hVel[i*4] = 0.0;
                        this.hVel[i*4 + 1] = 0.0;
                        this.hVel[i*4 + 2] = 0.0;
                        this.hVel[i*4 + 3] = 0.0;
                    }

                }
            }

        }
    }

    reset()
    {
        let jitter = particleRadius * 0.01;
        let s = Math.ceil(Math.pow(this.numParticles, 1.0/3.0));
        let gridSize = [s,s,s];
        this.initGrid(gridSize, particleRadius*2.0, jitter);

        engine_ctx.queue.writeBuffer(this.dPos, 0, this.hPos.buffer, 0, this.hPos.length * 4);
        engine_ctx.queue.writeBuffer(this.dVel, 0, this.hVel.buffer, 0, this.hVel.length * 4);
    }

    addSphere(pos, r)
    {
        let index = 0;
        let spacing = particleRadius*2.0;
        let jitter = particleRadius * 0.01;
        for (let z=-r; z<=r; z++)
        {
            for (let y=-r; y<=r; y++)
            {
                for (let x=-r; x<=r; x++)
                {
                    let dx = x*spacing;
                    let dy = y*spacing;
                    let dz = z*spacing;
                    let l = Math.sqrt(dx*dx + dy*dy + dz*dz);
                    if ((l <= particleRadius*2.0*r) && (index < this.numParticles))
                    {
                        this.hPos[index*4] = pos[0] + dx + (Math.random() * 2.0 - 1.0) * jitter;
                        this.hPos[index*4 + 1] = pos[1] + dy + (Math.random() * 2.0 - 1.0) * jitter;
                        this.hPos[index*4 + 2] = pos[2] + dz  + (Math.random() * 2.0 - 1.0) * jitter;
                        this.hPos[index*4 + 3] = 1.0;

                        this.hVel[index*4] = 0.0;
                        this.hVel[index*4 + 1] = 0.0;
                        this.hVel[index*4 + 2] = 0.0;
                        this.hVel[index*4 + 3] = 0.0;
                        index++;
                    }
                }
            }
        }

        engine_ctx.queue.writeBuffer(this.dPos, 0, this.hPos.buffer, 0, index * 4 * 4);
        engine_ctx.queue.writeBuffer(this.dVel, 0, this.hVel.buffer, 0, index * 4 * 4);
    }

    update(deltaTime)
    {
        {
            const uniform = new Int32Array(4);
            const funiform = new Float32Array(uniform.buffer);
            uniform[0] = this.numParticles;
            funiform[1] = deltaTime;
            engine_ctx.queue.writeBuffer(this.dConstant, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
        }

        {
            const uniform = new Int32Array(4);            
            uniform[0] = this.numGridCells;            
            engine_ctx.queue.writeBuffer(this.dConstant2, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
        }

        {
            const cell_count = new Uint32Array(this.numGridCells);        
            engine_ctx.queue.writeBuffer(this.dCellCount, 0, cell_count.buffer, cell_count.byteOffset, cell_count.byteLength);

            const group_count = new Uint32Array(1);    
            engine_ctx.queue.writeBuffer(this.buf_workgroup_counter, 0, group_count.buffer, group_count.byteOffset, group_count.byteLength);

            let num_groups = Math.floor((this.numGridCells + workgroup_size_2x - 1)/workgroup_size_2x);
            const group_state = new Uint32Array(num_groups*3);
            engine_ctx.queue.writeBuffer(this.buf_workgroup_state, 0, group_state.buffer, group_state.byteOffset, group_state.byteLength);
        }

        let commandEncoder = engine_ctx.device.createCommandEncoder();    
        Integrate(commandEncoder, this);
        HashCount(commandEncoder, this);
        PrefixSum(commandEncoder, this);
        Scatter(commandEncoder, this);
        Collide(commandEncoder, this);

        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);


    }


}

