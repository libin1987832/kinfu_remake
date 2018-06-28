#include "precomp.hpp"
#include <iostream>
#include <fstream>
using namespace std;


using namespace kfusion;
using namespace kfusion::cuda;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume::Entry

float kfusion::cuda::TsdfVolume::Entry::half2float(half)
{ throw "Not implemented"; }

kfusion::cuda::TsdfVolume::Entry::half kfusion::cuda::TsdfVolume::Entry::float2half(float value)
{ throw "Not implemented"; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// TsdfVolume

kfusion::cuda::TsdfVolume::TsdfVolume(const Vec3i& dims) : data_(), trunc_dist_(0.03f), max_weight_(128), dims_(dims),
    size_(Vec3f::all(3.f)), pose_(Affine3f::Identity()), gradient_delta_factor_(0.75f), raycast_step_factor_(0.75f)
{ create(dims_); }

kfusion::cuda::TsdfVolume::~TsdfVolume() {}

void kfusion::cuda::TsdfVolume::create(const Vec3i& dims)
{
    dims_ = dims;
    int voxels_number = dims_[0] * dims_[1] * dims_[2];
    data_.create(voxels_number * sizeof(int));
    setTruncDist(trunc_dist_);
    clear();
}

Vec3i kfusion::cuda::TsdfVolume::getDims() const
{ return dims_; }

Vec3f kfusion::cuda::TsdfVolume::getVoxelSize() const
{
    return Vec3f(size_[0]/dims_[0], size_[1]/dims_[1], size_[2]/dims_[2]);
}

const CudaData kfusion::cuda::TsdfVolume::data() const { return data_; }
CudaData kfusion::cuda::TsdfVolume::data() {  return data_; }
Vec3f kfusion::cuda::TsdfVolume::getSize() const { return size_; }

void kfusion::cuda::TsdfVolume::setSize(const Vec3f& size)
{ size_ = size; setTruncDist(trunc_dist_); }

float kfusion::cuda::TsdfVolume::getTruncDist() const { return trunc_dist_; }

void kfusion::cuda::TsdfVolume::setTruncDist(float distance)
{
    Vec3f vsz = getVoxelSize();
    float max_coeff = std::max<float>(std::max<float>(vsz[0], vsz[1]), vsz[2]);
    trunc_dist_ = std::max (distance, 2.1f * max_coeff);
}

int kfusion::cuda::TsdfVolume::getMaxWeight() const { return max_weight_; }
void kfusion::cuda::TsdfVolume::setMaxWeight(int weight) { max_weight_ = weight; }
Affine3f kfusion::cuda::TsdfVolume::getPose() const  { return pose_; }
void kfusion::cuda::TsdfVolume::setPose(const Affine3f& pose) { pose_ = pose; }
float kfusion::cuda::TsdfVolume::getRaycastStepFactor() const { return raycast_step_factor_; }
void kfusion::cuda::TsdfVolume::setRaycastStepFactor(float factor) { raycast_step_factor_ = factor; }
float kfusion::cuda::TsdfVolume::getGradientDeltaFactor() const { return gradient_delta_factor_; }
void kfusion::cuda::TsdfVolume::setGradientDeltaFactor(float factor) { gradient_delta_factor_ = factor; }
void kfusion::cuda::TsdfVolume::swap(CudaData& data) { data_.swap(data); }
void kfusion::cuda::TsdfVolume::applyAffine(const Affine3f& affine) { pose_ = affine * pose_; }

void kfusion::cuda::TsdfVolume::clear()
{ 
    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::clear_volume(volume);
}

void kfusion::cuda::TsdfVolume::integrate(const Dists& dists, const Affine3f& camera_pose, const Intr& intr)
{
    Affine3f vol2cam = camera_pose.inv() * pose_;

    device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff = device_cast<device::Aff3f>(vol2cam);

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::integrate(dists, volume, aff, proj);
}
void kfusion::cuda::TsdfVolume::integrateColor(const Dists& dists, const Affine3f& camera_pose, const Intr& intr, const kfusion::cuda::ImageRGB& image, cuda::ColorVolume& color_volume_)
{
	Affine3f vol2cam = camera_pose.inv() * pose_;

	device::Projector proj(intr.fx, intr.fy, intr.cx, intr.cy);

	device::Vec3i dims = device_cast<device::Vec3i>(dims_);
	device::Vec3f vsz = device_cast<device::Vec3f>(getVoxelSize());
	device::Aff3f aff = device_cast<device::Aff3f>(vol2cam);

	device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
	device::updateColorVolume(dists, volume, aff, proj,color_volume_.data(),image);
}
void kfusion::cuda::TsdfVolume::raycast(const Affine3f& camera_pose, const Intr& intr, Depth& depth, Normals& normals)
{
    DeviceArray2D<device::Normal>& n = (DeviceArray2D<device::Normal>&)normals;

    Affine3f cam2vol = pose_.inv() * camera_pose;

    device::Aff3f aff = device_cast<device::Aff3f>(cam2vol);
    device::Mat3f Rinv = device_cast<device::Mat3f>(cam2vol.rotation().inv(cv::DECOMP_SVD));

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, aff, Rinv, reproj, depth, n, raycast_step_factor_, gradient_delta_factor_);

}

void kfusion::cuda::TsdfVolume::raycast(const Affine3f& camera_pose, const Intr& intr, Cloud& points, Normals& normals)
{
    device::Normals& n = (device::Normals&)normals;
    device::Points& p = (device::Points&)points;

    Affine3f cam2vol = pose_.inv() * camera_pose;

    device::Aff3f aff = device_cast<device::Aff3f>(cam2vol);
    device::Mat3f Rinv = device_cast<device::Mat3f>(cam2vol.rotation().inv(cv::DECOMP_SVD));

    device::Reprojector reproj(intr.fx, intr.fy, intr.cx, intr.cy);

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());

    device::TsdfVolume volume(data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::raycast(volume, aff, Rinv, reproj, p, n, raycast_step_factor_, gradient_delta_factor_);
}

DeviceArray<Point> kfusion::cuda::TsdfVolume::fetchCloud(DeviceArray<Point>& cloud_buffer) const
{
    enum { DEFAULT_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000 };

    if (cloud_buffer.empty ())
        cloud_buffer.create (DEFAULT_CLOUD_BUFFER_SIZE);

    DeviceArray<device::Point>& b = (DeviceArray<device::Point>&)cloud_buffer;

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff  = device_cast<device::Aff3f>(pose_);

    device::TsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    size_t size = extractCloud(volume, aff, b);

    return DeviceArray<Point>((Point*)cloud_buffer.ptr(), size);
}



DeviceArray<Point> kfusion::cuda::TsdfVolume::fetchCloud2(DeviceArray<Point>& cloud_buffer) const
{
	enum { DEFAULT_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000 };

	if (cloud_buffer.empty())
		cloud_buffer.create(DEFAULT_CLOUD_BUFFER_SIZE);

	DeviceArray<device::Point>& b = (DeviceArray<device::Point>&)cloud_buffer;

	device::Vec3i dims = device_cast<device::Vec3i>(dims_);
	device::Vec3f vsz = device_cast<device::Vec3f>(getVoxelSize());
	device::Aff3f aff = device_cast<device::Aff3f>(pose_);

	device::TsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
	size_t size = extractCloud2(volume, aff, b);

	return DeviceArray<Point>((Point*)cloud_buffer.ptr(), size);
}

void kfusion::cuda::TsdfVolume::fetchNormals(const DeviceArray<Point>& cloud, DeviceArray<Normal>& normals) const
{
    normals.create(cloud.size());
    DeviceArray<device::Point>& c = (DeviceArray<device::Point>&)cloud;

    device::Vec3i dims = device_cast<device::Vec3i>(dims_);
    device::Vec3f vsz  = device_cast<device::Vec3f>(getVoxelSize());
    device::Aff3f aff  = device_cast<device::Aff3f>(pose_);
    device::Mat3f Rinv = device_cast<device::Mat3f>(pose_.rotation().inv(cv::DECOMP_SVD));

    device::TsdfVolume volume((ushort2*)data_.ptr<ushort2>(), dims, vsz, trunc_dist_, max_weight_);
    device::extractNormals(volume, c, aff, Rinv, gradient_delta_factor_, (float4*)normals.ptr());
}

float half_to_float(unsigned short h)
{
	short *ptr;
	int fs, fe, fm, rlt;

	ptr = (short *)&h;

	fs = ((*ptr) & 0x8000) << 16;

	fe = ((*ptr) & 0x7c00) >> 10;
	fe = fe + 0x70;
	fe = fe << 23;

	fm = ((*ptr) & 0x03ff) << 13;

	rlt = fs | fe | fm;
	return *((float *)&rlt);
}
void kfusion::cuda::TsdfVolume::save_volume(const std::string& fileName)
{
	int len = dims_[0] * dims_[1] * dims_[2];
	device::Vec3i dims = device_cast<device::Vec3i>(dims_);
	device::Vec3f vsz = device_cast<device::Vec3f>(getVoxelSize());
	//ushort2* aa= data_.ptr<ushort2>();
	//if (alen < 1)
	//	return;
	ushort2 * dataS = new ushort2[len];
	//float * dataf = new float[len];
	data_.download(dataS);
	std::ofstream out_file(fileName, std::ios::binary | std::ios::out);
	float t[8];
	t[0] = dims_[0]; t[1] = dims_[1]; t[2] = dims_[2]; t[3] = 0; t[4] = 0; t[5] = 0; t[6] = size_[0]; t[7] = trunc_dist_;
	out_file.write((char*)t, sizeof(float) * 8);
	
	
	float* dataf = new float[len];
	for (int j = 0; j < len; j++)
		dataf[j] = half_to_float(dataS[j].x);

	//float dataf[4];
	//for (int i = 0; i < dims_[0]; i++)
	//	for (int j = 0; j < dims_[1]; j++)
	//		for (int k = 0; k < dims_[2]; k++)
	//{
	//	//dataf[j] = half_to_float(dataS[j].x);
	//	dataf[3] = half_to_float(dataS[j].x);
	//	if (dataf[3]>-0.1 && dataf[3]<0.1)
	//	{
	//		dataf[0] = i; dataf[1] = j; dataf[2] = k; 
	//		out_file.write((char*)&dataf[0], sizeof(float)*4);
	//	}
	//	
	//}



	out_file.write((char*)&dataf[0], sizeof(float)*len);

	 out_file.close();
	 delete[] dataS;
     delete[] dataf;
}

void kfusion::cuda::TsdfVolume::save_ply(const std::string& fileName)
{
	int len = dims_[0] * dims_[1] * dims_[2];
	device::Vec3i dims = device_cast<device::Vec3i>(dims_);
	device::Vec3f vox_size = device_cast<device::Vec3f>(getVoxelSize());
	//ushort2* aa= data_.ptr<ushort2>();
	//if (alen < 1)
	//	return;
	ushort2 * dataS = new ushort2[len];

	data_.download(dataS);

	float tsdf_threshold = 0.2f;
	float weight_threshold = 1.0f;
	// float radius = 5.0f;

	// Count total number of points in point cloud
	int num_points = 0;
	for (int i = 0; i < dims.x * dims.y * dims.z; i++)
	{
		if (std::abs(half_to_float(dataS[i].x)) < tsdf_threshold && dataS[i].y > weight_threshold)
			num_points++;
		//if (dataS[i].y > 0){
			
		//	cout << half_to_float(dataS[i].x) << endl;
		//}
	}

	// Create header for ply file
	FILE *fp = fopen(fileName.c_str(), "w");
	fprintf(fp, "ply\n");
	fprintf(fp, "format binary_little_endian 1.0\n");
	fprintf(fp, "element vertex %d\n", num_points);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "end_header\n");

	// Create point cloud content for ply file
	for (int i = 0; i < dims.x * dims.y * dims.z; i++) {

		// If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
		if (std::abs(half_to_float(dataS[i].x)) < tsdf_threshold && dataS[i].y > weight_threshold) {

			// Compute voxel indices in int for higher positive number range
			int z = floor(i / (vox_size.x * vox_size.y));
			int y = floor((i - (z * vox_size.x * vox_size.y)) / vox_size.x);
			int x = i - (z * vox_size.x * vox_size.y) - (y * vox_size.y);

			// Convert voxel indices to float, and save coordinates to ply file
			float float_x = (float)x;
			float float_y = (float)y;
			float float_z = (float)z;
			fwrite(&float_x, sizeof(float), 1, fp);
			fwrite(&float_y, sizeof(float), 1, fp);
			fwrite(&float_z, sizeof(float), 1, fp);
		}
	}
	fclose(fp);
	delete[] dataS;
}
void save_volume_to_ply(const std::string &file_name, int* vox_size, float* vox_tsdf, float* vox_weight) {
	float tsdf_threshold = 0.2f;
	float weight_threshold = 1.0f;
	// float radius = 5.0f;

	// Count total number of points in point cloud
	int num_points = 0;
	for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++)
		if (std::abs(vox_tsdf[i]) < tsdf_threshold && vox_weight[i] > weight_threshold)
			num_points++;

	// Create header for ply file
	FILE *fp = fopen(file_name.c_str(), "w");
	fprintf(fp, "ply\n");
	fprintf(fp, "format binary_little_endian 1.0\n");
	fprintf(fp, "element vertex %d\n", num_points);
	fprintf(fp, "property float x\n");
	fprintf(fp, "property float y\n");
	fprintf(fp, "property float z\n");
	fprintf(fp, "end_header\n");

	// Create point cloud content for ply file
	for (int i = 0; i < vox_size[0] * vox_size[1] * vox_size[2]; i++) {

		// If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
		if (std::abs(vox_tsdf[i]) < tsdf_threshold && vox_weight[i] > weight_threshold) {

			// Compute voxel indices in int for higher positive number range
			int z = floor(i / (vox_size[0] * vox_size[1]));
			int y = floor((i - (z * vox_size[0] * vox_size[1])) / vox_size[0]);
			int x = i - (z * vox_size[0] * vox_size[1]) - (y * vox_size[0]);

			// Convert voxel indices to float, and save coordinates to ply file
			float float_x = (float)x;
			float float_y = (float)y;
			float float_z = (float)z;
			fwrite(&float_x, sizeof(float), 1, fp);
			fwrite(&float_y, sizeof(float), 1, fp);
			fwrite(&float_z, sizeof(float), 1, fp);
		}
	}
	fclose(fp);
}