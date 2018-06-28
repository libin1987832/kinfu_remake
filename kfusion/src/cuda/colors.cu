/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "device.hpp"
#include <kfusion/types.hpp>
//#include "pcl/gpu/utils/device/vector_math.hpp"
#include "texture_binder.hpp"
namespace kfusion
{
  namespace device
  {

    __global__ void
		initColorVolumeKernel(PtrStep<uchar4> volume, int VOLUME_X, int VOLUME_Y, int VOLUME_Z)
    {
      int x = threadIdx.x + blockIdx.x * blockDim.x;
      int y = threadIdx.y + blockIdx.y * blockDim.y;

      if (x < VOLUME_X && y < VOLUME_Y)
      {
        uchar4 *pos = volume.ptr (y) + x;
        int z_step = VOLUME_Y * volume.step / sizeof(*pos);

#pragma unroll
        for (int z = 0; z < VOLUME_Z; ++z, pos += z_step)
          *pos = make_uchar4 (0, 0, 0, 0);
      }
    }
  }
}

void
kfusion::device::initColorVolume(PtrStep<uchar4> color_volume, int V_X,int V_Y,int V_Z)
{
  dim3 block (32, 16);
  dim3 grid (1, 1, 1);
  grid.x = divUp(V_X, block.x);
  grid.y = divUp(V_Y, block.y);

  initColorVolumeKernel<<<grid, block>>>(color_volume,V_X,V_Y,V_Z);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace kfusion
{
  namespace device
  {
   texture<float, 2> color_tex(0, cudaFilterModePoint, cudaAddressModeBorder, cudaCreateChannelDescHalf());

    struct ColorVolumeImpl
    {

      mutable PtrStep<uchar4> color_volume;
	      Aff3f vol2cam;
            Projector proj;
            int2 dists_size;

            float tranc_dist_inv;
			    PtrStepSz<uchar3> colors;
				Intr intr;
            __kf_device__
            void operator()(TsdfVolume& volume) const
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= volume.dims.x || y >= volume.dims.y)
                    return;

                //float3 zstep = vol2cam.R * make_float3(0.f, 0.f, volume.voxel_size.z);
                float3 zstep = make_float3(vol2cam.R.data[0].z, vol2cam.R.data[1].z, vol2cam.R.data[2].z) * volume.voxel_size.z;

                float3 vx = make_float3(x * volume.voxel_size.x, y * volume.voxel_size.y, 0);
                float3 vc = vol2cam * vx; //tranform from volume coo frame to camera one

                TsdfVolume::elem_type* vptr = volume.beg(x, y);
                for(int i = 0; i < volume.dims.z; ++i, vc += zstep, vptr = volume.zstep(vptr))
                {
                    float2 coo = proj(vc);

                    //#if defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 300
                    // this is actually workaround for kepler. it doesn't return 0.f for texture
                    // fetches for out-of-border coordinates even for cudaaddressmodeborder mode
                    if (coo.x < 0 || coo.y < 0 || coo.x >= dists_size.x || coo.y >= dists_size.y)
                        continue;
                    //#endif
                    float Dp = tex2D(color_tex, coo.x, coo.y);
                    if(Dp == 0 || vc.z <= 0)
                        continue;

                    float sdf = Dp - __fsqrt_rn(dot(vc, vc)); //Dp - norm(v)

                    if (sdf >= -volume.trunc_dist)
                    {
            //if (update)
						{
							uchar4 *ptr = color_volume.ptr(volume.dims.y * i + y) + x;
							uchar3 rgb = colors.ptr (__float2int_rn(coo.y))[__float2int_rn(coo.x)];
							uchar4 volume_rgbw = *ptr;

							int weight_prev = volume_rgbw.w;

							const float Wrk = 1.f;
							float new_x = (volume_rgbw.x * weight_prev + Wrk * rgb.x) / (weight_prev + Wrk);
							float new_y = (volume_rgbw.y * weight_prev + Wrk * rgb.y) / (weight_prev + Wrk);
							float new_z = (volume_rgbw.z * weight_prev + Wrk * rgb.z) / (weight_prev + Wrk);

							int weight_new = weight_prev + 1;

							uchar4 volume_rgbw_new;
							volume_rgbw_new.x = min (255, max (0, __float2int_rn (new_x)));
							volume_rgbw_new.y = min (255, max (0, __float2int_rn (new_y)));
							volume_rgbw_new.z = min (255, max (0, __float2int_rn (new_z)));
							volume_rgbw_new.w = min (volume.max_weight, weight_new);

							*ptr = volume_rgbw_new;
						}
                    }
                }  // for(;;)
            }

    };

    __global__ void
    updateColorVolumeKernel (const ColorVolumeImpl cvi,TsdfVolume volume) {
      cvi (volume);
    }
  }
}

void
kfusion::device::updateColorVolume(const PtrStepSz<ushort>& dists, TsdfVolume& volume, const Aff3f& aff, const Projector& proj, PtrStep<uchar4> color_volume,const PtrStepSz<uchar3>& colors)
{
  ColorVolumeImpl ti;
	ti.dists_size = make_int2(dists.cols, dists.rows);
    ti.vol2cam = aff;
    ti.proj = proj;
    ti.tranc_dist_inv = 1.f/volume.trunc_dist;
	ti.color_volume=color_volume;
	ti.colors=colors;


	 color_tex.filterMode = cudaFilterModePoint;
    color_tex.addressMode[0] = cudaAddressModeBorder;
    color_tex.addressMode[1] = cudaAddressModeBorder;
    color_tex.addressMode[2] = cudaAddressModeBorder;
    TextureBinder binder(dists, color_tex, cudaCreateChannelDescHalf()); (void)binder;

   dim3 block(32, 8);
    dim3 grid(divUp(volume.dims.x, block.x), divUp(volume.dims.y, block.y));

  updateColorVolumeKernel<<<grid, block>>>(ti,volume);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
}

namespace kfusion
{
  namespace device
  {
    __global__ void
		extractColorsKernel(const float3 cell_size, const PtrStep<uchar4> color_volume, const PtrSz<Point> points, uchar4 *colors, int VOLUME_Y)
    {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;

      if (idx < points.size)
      {
        int3 v;
        float3 p = *(const float3*)(points.data + idx);
        v.x = __float2int_rd (p.x / cell_size.x);        // round to negative infinity
        v.y = __float2int_rd (p.y / cell_size.y);
        v.z = __float2int_rd (p.z / cell_size.z);
		//if (VOLUME_Y * v.z + v.y < 0)
	//		printf("error1:%d %d %d %d", VOLUME_Y * v.z + v.y,v.x,v.y,v.z);
		//else
		//	printf("error2:%d %d %d %d", VOLUME_Y * v.z + v.y, v.x, v.y, v.z);
        uchar4 rgbw = color_volume.ptr (VOLUME_Y * v.z + v.y)[v.x];
        colors[idx] = make_uchar4 (rgbw.z, rgbw.y, rgbw.x, 0); //bgra
      }
    }
  }
}

void
kfusion::device::exctractColors(const PtrStep<uchar4>& color_volume, const float3& volume_size, const PtrSz<Point>& points, uchar4* colors, int VOLUME_X, int VOLUME_Y, int VOLUME_Z)
{
  const int block = 256;
  float3 cell_size = make_float3 (volume_size.x / VOLUME_X, volume_size.y / VOLUME_Y, volume_size.z / VOLUME_Z);
  extractColorsKernel << <divUp(points.size, block), block >> >(cell_size, color_volume, points, colors, VOLUME_Y);
  cudaSafeCall ( cudaGetLastError () );
  cudaSafeCall (cudaDeviceSynchronize ());
};
