import { Integrate } from "./integrate.js"
import { HashCount } from "./hash_count.js"
import { PrefixSum } from "./prefix_sum_m.js"
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

function getRandomInt(max) 
{
    return Math.floor(Math.random() * max);
}

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

        this.dCellCountBufs = [];
        this.dCellCountBufSizes = [];

        let buf_size = this.numGridCells;
        while (buf_size>0)
        {
            let buf = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);    
            this.dCellCountBufs.push(buf);
            this.dCellCountBufSizes.push(buf_size);
            buf_size = Math.floor((buf_size + workgroup_size_2x - 1)/workgroup_size_2x) - 1;
        }        
        
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

        let layout_entries_integrate = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 1,
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
                    type: "read-only-storage"
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
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },        
        ];

        let bindGroupLayoutHashCount = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_hash_count });
        engine_ctx.cache.bindGroupLayouts.hash_count = bindGroupLayoutHashCount;

        let layout_entries_prefix_sum1 = [ 
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            }
        ];

        let bindGroupLayoutPrefixSum1A = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_prefix_sum1 });
        engine_ctx.cache.bindGroupLayouts.prefix_sum_1a = bindGroupLayoutPrefixSum1A;

        layout_entries_prefix_sum1.push({
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        });
    
        let bindGroupLayoutPrefixSum1B = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_prefix_sum1 });
        engine_ctx.cache.bindGroupLayouts.prefix_sum_1b = bindGroupLayoutPrefixSum1B;

        let layout_entries_prefix_sum2 = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
                }
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
                }
            }
        ];
    
        let bindGroupLayoutPrefixSum2 = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_prefix_sum2 });
        engine_ctx.cache.bindGroupLayouts.prefix_sum_2 = bindGroupLayoutPrefixSum2;
    


        let layout_entries_scatter = [ 
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "read-only-storage"
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
                    type: "storage"
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

        ];

        let bindGroupLayoutScatter = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_scatter });
        engine_ctx.cache.bindGroupLayouts.scatter = bindGroupLayoutScatter;

        let layout_entries_collide = [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer:{
                    type: "storage"
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

        ];

        let bindGroupLayoutCollide = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_collide });
        engine_ctx.cache.bindGroupLayouts.collide = bindGroupLayoutCollide;

        let group_entries_integrate = [
            {
                binding: 0,
                resource:{
                    buffer: this.dPos            
                }
            },
            {
                binding: 1,
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
                    buffer: this.dPos            
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dGridParticleHash
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dGridParticleIndexInCell
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dCellCountBufs[0]
                }
            },

        ];

        this.bind_group_hash_count = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutHashCount, entries: group_entries_hash_count});

        this.bind_group_prefix_sum1 = [];
        this.bind_group_prefix_sum2 = [];

        for (let i=0; i<this.dCellCountBufs.length; i++)
        {
            if (i<this.dCellCountBufs.length - 1)
            {
                let group_entries = [ 
                    {
                        binding: 0,
                        resource:{
                            buffer: this.dCellCountBufs[i]            
                        }
                    },
                    {
                        binding: 1,
                        resource:{
                            buffer: this.dCellCountBufs[i+1]
                        }
                    }       
                ];
                {
                    let bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutPrefixSum1B, entries: group_entries});
                    this.bind_group_prefix_sum1.push(bind_group);
                }
                {
                    let bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutPrefixSum2, entries: group_entries});
                    this.bind_group_prefix_sum2.push(bind_group);
                }
            }
            else if (this.dCellCountBufSizes[i] > 1)
            {
                let group_entries = [ 
                    {
                        binding: 0,
                        resource:{
                            buffer: this.dCellCountBufs[i]            
                        }
                    }              
                ];
                let bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutPrefixSum1A, entries: group_entries});
                this.bind_group_prefix_sum1.push(bind_group);
            }

        }
    
        let group_entries_scatter = [
            {
                binding: 0,
                resource:{
                    buffer: this.dPos            
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
                    buffer: this.dCellCountBufs[0]            
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dGridParticleHash            
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dGridParticleIndexInCell            
                }
            },
            {
                binding: 5,
                resource:{
                    buffer: this.dSortedPos            
                }
            },
            {
                binding: 6,
                resource:{
                    buffer: this.dSortedVel            
                }
            },
            {
                binding: 7,
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
                    buffer: this.dVel            
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: this.dSortedPos            
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: this.dSortedVel            
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: this.dGridParticleIndex            
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: this.dCellCountBufs[0]            
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

    default()
    {
        if (!("frame" in this))
        {
            this.frame = 0;
        }

        if (this.frame++>1000)
        {
            let ballr = 10 + getRandomInt(10);
            let tr = particleRadius + particleRadius * 2.0  * ballr;
            let pos = [ -1.0 + tr + Math.random() *(2.0 - tr*2.0), 1.0 - tr,  -1.0 + tr + Math.random() *(2.0 - tr*2.0)];
            this.addSphere(pos, ballr);
            this.frame = 0;
        }
    }

    flow()
    {        
        if (!("flow_pos" in this))
        {
            this.flow_pos = 0;
            this.flow_frame = 0;
        }

        let length = 16;
        let count = length*length;
        let rate = 5;
        let speed = particleRadius*2.0 / 0.5 /rate;

        if (this.flow_frame==0)
        {
            for (let i=0; i<length; i++)
            {
                for (let j=0; j<length; j++)
                {
                    let index = this.flow_pos + j + i*length;

                    this.hPos[index*4] = -1.0 + particleRadius;
                    this.hPos[index*4 + 1] = 1.0 - particleRadius - i * particleRadius * 2.0;
                    this.hPos[index*4 + 2] = (j-(length-1)*0.5) * particleRadius * 2.0;
                    this.hPos[index*4 + 3] = 1.0;

                    this.hVel[index*4] = speed;
                    this.hVel[index*4 + 1] = 0.0;
                    this.hVel[index*4 + 2] = 0.0;
                    this.hVel[index*4 + 3] = 0.0;
                }
            }

            engine_ctx.queue.writeBuffer(this.dPos, this.flow_pos*4*4, this.hPos.buffer, this.flow_pos*4*4, count*4*4);
            engine_ctx.queue.writeBuffer(this.dVel, this.flow_pos*4*4, this.hVel.buffer, this.flow_pos*4*4, count*4*4);        

            this.flow_pos = (this.flow_pos + count) % this.numParticles;
        }
        this.flow_frame = (this.flow_frame + 1) % rate;

    }

    flow2()
    {
        if (!("flow_pos" in this))
        {
            this.flow_pos = 0;
            this.flow_frame = 0;
        }

        let count = 256;
        let radius = Math.sqrt(count / (Math.PI)) *  particleRadius * 2.0;
        let rate = 5;
        let speed = particleRadius*2.0 / 0.5 /rate;


        if (this.flow_frame==0)
        {
            let jitter = Math.random();
            for (let i=0; i< count; i++)
            {
                let index = this.flow_pos + i;

                let r = Math.sqrt((i+0.5)/count) * radius;
                let theta = i * 2.4 + jitter * 2.0 * Math.PI;     

                let x = r * Math.cos(theta);
                let y = r * Math.sin(theta);

                this.hPos[index*4] = -1.0 + particleRadius;
                this.hPos[index*4 + 1] = 1.0 - particleRadius - radius + y;
                this.hPos[index*4 + 2] = x;
                this.hPos[index*4 + 3] = 1.0;

                this.hVel[index*4] = speed;
                this.hVel[index*4 + 1] = 0.0;
                this.hVel[index*4 + 2] = 0.0;
                this.hVel[index*4 + 3] = 0.0;

            }

            engine_ctx.queue.writeBuffer(this.dPos, this.flow_pos*4*4, this.hPos.buffer, this.flow_pos*4*4, count*4*4);
            engine_ctx.queue.writeBuffer(this.dVel, this.flow_pos*4*4, this.hVel.buffer, this.flow_pos*4*4, count*4*4);

            this.flow_pos = (this.flow_pos + count) % this.numParticles;
        }
        this.flow_frame = (this.flow_frame + 1) % rate;
    }

    update()
    {
        {
            const cell_count = new Uint32Array(this.numGridCells);        
            engine_ctx.queue.writeBuffer(this.dCellCountBufs[0], 0, cell_count.buffer, cell_count.byteOffset, cell_count.byteLength);            
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

