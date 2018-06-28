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

#include <color_volume.h>
#include "precomp.hpp"
#include <algorithm>
//#include <Eigen/Core>
#include <math.h>
#include <fstream>

using kfusion::device::device_cast;
using namespace kfusion;
using namespace kfusion::cuda;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kfusion::cuda::ColorVolume::ColorVolume(const TsdfVolume& tsdf, int max_weight) : resolution_(tsdf.getDims()), volume_size_(tsdf.getSize()), max_weight_(1)
{
  max_weight_ = max_weight < 0 ? max_weight_ : max_weight;
  max_weight_ = max_weight_ > 255 ? 255 : max_weight_;

  int volume_x = resolution_(0);
  int volume_y = resolution_(1);
  int volume_z = resolution_(2);

  color_volume_.create (volume_y * volume_z, volume_x);
  reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kfusion::cuda::ColorVolume::~ColorVolume()
{

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
kfusion::cuda::ColorVolume::reset()
{
	device::initColorVolume(color_volume_, resolution_[0], resolution_[1], resolution_[2]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
kfusion::cuda::ColorVolume::getMaxWeight() const
{
  return max_weight_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

DeviceArray2D<int>
kfusion::cuda::ColorVolume::data() const
{
  return color_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
kfusion::cuda::ColorVolume::fetchColors(const DeviceArray<PointType>& cloud, DeviceArray<RGB>& colors) const
{  
  colors.create(cloud.size());
  device::exctractColors(color_volume_, device_cast<const float3> (volume_size_), cloud, (uchar4*)colors.ptr()/*bgra*/, resolution_[0], resolution_[1], resolution_[2]);
}
void kfusion::cuda::ColorVolume::save_volume(const TsdfVolume& tsdf,const std::string& fileName)
{
	int len = resolution_[0] * resolution_[1] * resolution_[2];
	device::Vec3i dims = device_cast<device::Vec3i>(resolution_);
	device::Vec3f vsz = device_cast<device::Vec3f>(volume_size_);

	DeviceArray<Point> cloud_buffer_device_;
	DeviceArray<Point> extracted = tsdf.fetchCloud(cloud_buffer_device_);
	cv::Mat cloud_host(1, (int)extracted.size(), CV_32FC4);
	extracted.download(cloud_host.ptr<Point>());
	DeviceArray<RGB> color_device;
	fetchColors(extracted, color_device);
	//ushort2* aa= data_.ptr<ushort2>();
	//if (alen < 1)
	//	return;
	vector<int>  dataS ;
	dataS.resize(resolution_[0] * resolution_[1] * resolution_[2]);
	vector<RGB>  dataf;
	//float * dataf = new float[len];
	color_device.download(dataf);
	std::ofstream out_file(fileName, std::ios::binary | std::ios::out);
	float t[8];
	t[0] = resolution_[0]; t[1] = resolution_[1]; t[2] = resolution_[2]; t[3] = 0; t[4] = 0; t[5] = 0; t[6] = volume_size_[0]; t[7] = 0;
	out_file.write((char*)t, sizeof(float) * 8);
	
	for (int i = 0; i < cloud_host.cols; i++)
	{
		cv::Vec4f t=cloud_host.at<cv::Vec4f>(i);
		int index = std::floorl(t[2]) * resolution_[0] * resolution_[1] + std::floorl(t[0]) * resolution_[1] + std::floorl(t[1]);
		dataS[index] = dataf[i].bgra;
	}
	//float* dataf = new float[len];
	//for (int j = 0; j < len; j++)
	//	dataf[j] = half_to_float(dataS[j].x);


	out_file.write((char*)&dataS[0], sizeof(int)*dataS.size());

	out_file.close();
	//delete[] dataS;
	//delete[] dataf;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////