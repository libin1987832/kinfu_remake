#pragma warning (disable :4996)
#undef _CRT_SECURE_NO_DEPRECATE
#include "XnCppWrapper.h"
#include <io/capture.hpp>
#include<opencv2\opencv.hpp> 
using namespace std;
using namespace xn;

//const std::string XMLConfig =
//"<OpenNI>"
//        "<Licenses>"
//        "<License vendor=\"PrimeSense\" key=\"0KOIk2JeIBYClPWVnMoRKn5cdY4=\"/>"
//        "</Licenses>"
//        "<Log writeToConsole=\"false\" writeToFile=\"false\">"
//                "<LogLevel value=\"3\"/>"
//                "<Masks>"
//                        "<Mask name=\"ALL\" on=\"true\"/>"
//                "</Masks>"
//                "<Dumps>"
//                "</Dumps>"
//        "</Log>"
//        "<ProductionNodes>"
//                "<Node type=\"Image\" name=\"Image1\">"
//                        "<Configuration>"
//                                "<MapOutputMode xRes=\"640\" yRes=\"480\" FPS=\"30\"/>"
//                                "<Mirror on=\"false\"/>"
//                        "</Configuration>"
//                "</Node> "
//                "<Node type=\"Depth\" name=\"Depth1\">"
//                        "<Configuration>"
//                                "<MapOutputMode xRes=\"640\" yRes=\"480\" FPS=\"30\"/>"
//                                "<Mirror on=\"false\"/>"
//                        "</Configuration>"
//                "</Node>"
//        "</ProductionNodes>"
//"</OpenNI>";

#define REPORT_ERROR(msg) kfusion::cuda::error ((msg), __FILE__, __LINE__)

struct kfusion::OpenNISource::Impl
{
    Context context;
    ScriptNode scriptNode;
    DepthGenerator depth;
    ImageGenerator image;
    ProductionNode node;
    DepthMetaData depthMD;
    ImageMetaData imageMD;
    XnChar strError[1024];
    Player player_;

    bool has_depth;
    bool has_image;
};

kfusion::OpenNISource::OpenNISource() : depth_focal_length_VGA (0.f), baseline (0.f),
    shadow_value (0), no_sample_value (0), pixelSize (0.0), max_depth (0) {}

kfusion::OpenNISource::OpenNISource(int device) {open (device); }
kfusion::OpenNISource::OpenNISource(const string& filename, bool repeat /*= false*/,TYPE type) {open (filename, repeat,type); }
kfusion::OpenNISource::~OpenNISource() { release (); }
//HRESULT kfusion::OpenNISource::ToggleNearMode()
//{
//	HRESULT hr = E_FAIL;
//
//	if (m_pNuiSensor)
//	{
//		hr = m_pNuiSensor->NuiImageStreamSetImageFrameFlags(m_pDepthStreamHandle, m_bNearMode ? 0 : NUI_IMAGE_STREAM_FLAG_ENABLE_NEAR_MODE);
//
//		if (SUCCEEDED(hr))
//		{
//			m_bNearMode = !m_bNearMode;
//		}
//	}
//
//	return hr;
//}
void kfusion::OpenNISource::open(int device)
{



	//INuiSensor * pNuiSensor = NULL;
	//HRESULT hr;

	//int iSensorCount = 0;
	//hr = NuiGetSensorCount(&iSensorCount);
	//if (FAILED(hr)) { return hr; }

	//// Look at each Kinect sensor
	//for (int i = 0; i < iSensorCount; ++i)
	//{
	//	// Create the sensor so we can check status, if we can't create it, move on to the next
	//	hr = NuiCreateSensorByIndex(i, &pNuiSensor);
	//	if (FAILED(hr))
	//	{
	//		continue;
	//	}

	//	// Get the status of the sensor, and if connected, then we can initialize it
	//	hr = pNuiSensor->NuiStatus();
	//	if (S_OK == hr)
	//	{
	//		m_pNuiSensor = pNuiSensor;
	//		break;
	//	}

	//	// This sensor wasn't OK, so release it since we're not using it
	//	pNuiSensor->Release();
	//}

	//if (NULL == m_pNuiSensor)
	//{
	//	return E_FAIL;
	//}

	//// Initialize the Kinect and specify that we'll be using depth
	//hr = m_pNuiSensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR | NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX);
	//if (FAILED(hr)) { return hr; }

	//// Create an event that will be signaled when depth data is available
	//m_hNextDepthFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	//// Open a depth image stream to receive depth frames
	//hr = m_pNuiSensor->NuiImageStreamOpen(
	//	NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX,
	//	cDepthResolution,
	//	0,
	//	2,
	//	m_hNextDepthFrameEvent,
	//	&m_pDepthStreamHandle);
	//if (FAILED(hr)) { return hr; }

	//// Create an event that will be signaled when color data is available
	//m_hNextColorFrameEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	//// Open a color image stream to receive color frames
	//hr = m_pNuiSensor->NuiImageStreamOpen(
	//	NUI_IMAGE_TYPE_COLOR,
	//	cColorResolution,
	//	0,
	//	2,
	//	m_hNextColorFrameEvent,
	//	&m_pColorStreamHandle);
	//if (FAILED(hr)) { return hr; }

	//// Start with near mode on
	//ToggleNearMode();

    impl_ = cv::Ptr<Impl>( new Impl () );

    XnMapOutputMode mode;
    mode.nXRes = XN_VGA_X_RES;
    mode.nYRes = XN_VGA_Y_RES;
    mode.nFPS = 30;

    XnStatus rc;
    rc = impl_->context.Init ();
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "Init failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }

    xn::NodeInfoList devicesList;
    rc = impl_->context.EnumerateProductionTrees ( XN_NODE_TYPE_DEVICE, NULL, devicesList, 0 );
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "Init failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }

    xn::NodeInfoList::Iterator it = devicesList.Begin ();
    for (int i = 0; i < device; ++i)
        it++;

    NodeInfo node = *it;
    rc = impl_->context.CreateProductionTree ( node, impl_->node );
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "Init failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }

    XnLicense license;
    const char* vendor = "PrimeSense";
    const char* key = "0KOIk2JeIBYClPWVnMoRKn5cdY4=";
    sprintf (license.strKey, key);
    sprintf (license.strVendor, vendor);

    rc = impl_->context.AddLicense (license);
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "licence failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }

    rc = impl_->depth.Create (impl_->context);
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "Depth generator  failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }
    //rc = impl_->depth.SetIntProperty("HoleFilter", 1);
    rc = impl_->depth.SetMapOutputMode (mode);
    impl_->has_depth = true;

    rc = impl_->image.Create (impl_->context);
    if (rc != XN_STATUS_OK)
    {
        printf ("Image generator creation failed: %s\n", xnGetStatusString (rc));
        impl_->has_image = false;
    }
    else
    {
        impl_->has_image = true;
        rc = impl_->image.SetMapOutputMode (mode);
    }

    getParams ();

    rc = impl_->context.StartGeneratingAll ();
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "Start failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }
}
void getFiles(string path, vector<string>& depthfiles, vector<string>& rgbfiles)
{
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR)) {  //比较文件类型是否是文件夹

			}
			else {
				rgbfiles.push_back(p.assign(path).append("\\").append(fileinfo.name));
				std::string rgb(fileinfo.name);
				char namergb[100] = { 0 };
				for (int i = 0; i < rgb.size() - 8; i++)
					namergb[i] = rgb[i];
				depthfiles.push_back(p.assign("F:\\depth").append("\\").append(namergb).append("depth.tiff"));
			}

		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
void kfusion::OpenNISource::open(const std::string& filename, bool repeat /*= false*/,TYPE type)
{
	fileType = type;
	if (fileType == FIFE_PICTURE)
	{
		//m_current_frame = 1570;
		m_current_frame = 0;
		return;
	}

    impl_ = cv::Ptr<Impl> ( new Impl () );

    XnStatus rc;

    rc = impl_->context.Init ();
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "Init failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }

    //rc = impl_->context.OpenFileRecording (filename.c_str (), impl_->node);
    rc = impl_->context.OpenFileRecording (filename.c_str (), impl_->player_);
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "Open failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }
    
    impl_->player_.SetRepeat(repeat);

    rc = impl_->context.FindExistingNode (XN_NODE_TYPE_DEPTH, impl_->depth);
    impl_->has_depth = (rc == XN_STATUS_OK);

    rc = impl_->context.FindExistingNode (XN_NODE_TYPE_IMAGE, impl_->image);
    impl_->has_image = (rc == XN_STATUS_OK);

    if (!impl_->has_depth)
        REPORT_ERROR ("No depth nodes. Check your configuration");

    if (impl_->has_depth)
        impl_->depth.GetMetaData (impl_->depthMD);

    if (impl_->has_image)
        impl_->image.GetMetaData (impl_->imageMD);

    // RGB is the only image format supported.
    if (impl_->imageMD.PixelFormat () != XN_PIXEL_FORMAT_RGB24)
        REPORT_ERROR ("Image format must be RGB24\n");

    getParams ();
}

void kfusion::OpenNISource::release ()
{
	if (fileType == FIFE_PICTURE)
		m_current_frame = 0;
    if (impl_)
    {
        impl_->context.StopGeneratingAll ();
        impl_->context.Release ();
    }

    impl_.release();;
    depth_focal_length_VGA = 0;
    baseline = 0.f;
    shadow_value = 0;
    no_sample_value = 0;
    pixelSize = 0.0;
	//if (NULL != m_pNuiSensor)
	//{
	//	m_pNuiSensor->NuiShutdown();
	//	m_pNuiSensor->Release();
	//}
	//CloseHandle(m_hNextDepthFrameEvent);
	//CloseHandle(m_hNextColorFrameEvent);
}

bool kfusion::OpenNISource::grab(cv::Mat& depth, cv::Mat& image)
{
	if (FIFE_PICTURE == fileType)
	{
		char index_C[10] = { 0 };
		//sprintf_s(index_C, "%05d", m_current_frame);
		sprintf_s(index_C, "%d", m_current_frame);
		std::string depth_name;
		cv::Mat pDepth = cv::imread(depth_name.assign("F://depth//frame_").append(index_C).append("_depth.tiff"), cv::IMREAD_ANYDEPTH);
		cv::Mat(pDepth.rows, pDepth.cols, CV_16U, pDepth.data).copyTo(depth);
		for (int i = 0; i < pDepth.rows; i++)
			for (int j = 0;j< pDepth.cols; j++)
		{
			if (depth.at<unsigned short>(i, j)>7000)
				depth.at<unsigned short>(i, j) = 0;
		}
		std::string color_name;
		image = cv::imread(color_name.assign("F://color//frame_").append(index_C).append("_rgb.tiff"));
		m_current_frame++;
		return 1;
	}


	//NUI_IMAGE_FRAME imageFrame;

	//HRESULT hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pDepthStreamHandle, 0, &imageFrame);
	//if (FAILED(hr)) { return hr; }

	//NUI_LOCKED_RECT LockedRect;
	//hr = imageFrame.pFrameTexture->LockRect(0, &LockedRect, NULL, 0);
	//if (FAILED(hr)) { return hr; }

	//cv::Mat(480, 640, CV_16U).copyTo(depth);

	//memcpy(depth.data, LockedRect.pBits, LockedRect.size);
	//m_bDepthReceived = true;

	//hr = imageFrame.pFrameTexture->UnlockRect(0);
	//if (FAILED(hr)) { return hr; };

	//hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_pDepthStreamHandle, &imageFrame);



	//NUI_IMAGE_FRAME imageFrame2;
	//NUI_LOCKED_RECT LockedRect2;
	// hr = m_pNuiSensor->NuiImageStreamGetNextFrame(m_pColorStreamHandle, 0, &imageFrame2);
	//if (FAILED(hr)) { return hr; }

	//hr = imageFrame2.pFrameTexture->LockRect(0, &LockedRect2, NULL, 0);
	//if (FAILED(hr)) { return hr; }
	//cv::Mat(480, 640, CV_8UC4).copyTo(image);
	//memcpy(image.data, LockedRect2.pBits, LockedRect2.size);
	//m_bColorReceived = true;

	//hr = imageFrame2.pFrameTexture->UnlockRect(0);
	//if (FAILED(hr)) { return hr; };

	//hr = m_pNuiSensor->NuiImageStreamReleaseFrame(m_pColorStreamHandle, &imageFrame2);




	XnStatus rc = XN_STATUS_OK;

	rc = impl_->context.WaitAndUpdateAll();
	if (rc != XN_STATUS_OK)
		return printf("Read failed: %s\n", xnGetStatusString(rc)), false;

	if (impl_->has_depth)
	{
		impl_->depth.GetMetaData(impl_->depthMD);
		const XnDepthPixel* pDepth = impl_->depthMD.Data();
		int x = impl_->depthMD.FullXRes();
		int y = impl_->depthMD.FullYRes();
		cv::Mat(y, x, CV_16U, (void*)pDepth).copyTo(depth);
	}
	else
	{
		depth.release();
		printf("no depth\n");
	}

	if (impl_->has_image)
	{
		impl_->image.GetMetaData(impl_->imageMD);
		const XnRGB24Pixel* pImage = impl_->imageMD.RGB24Data();
		int x = impl_->imageMD.FullXRes();
		int y = impl_->imageMD.FullYRes();
		image.create(y, x, CV_8UC3);

		cv::Vec3b *dptr = image.ptr<cv::Vec3b>();
		for (size_t i = 0; i < image.total(); ++i)
			dptr[i] = cv::Vec3b(pImage[i].nBlue, pImage[i].nGreen, pImage[i].nRed);
	}
	else
	{
		image.release();
		printf("no image\n");
	}

	return impl_->has_image || impl_->has_depth;
}

void kfusion::OpenNISource::getParams ()
{
    XnStatus rc = XN_STATUS_OK;

    max_depth = impl_->depth.GetDeviceMaxDepth ();

    rc = impl_->depth.GetRealProperty ( "ZPPS", pixelSize );  // in mm
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "ZPPS failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }

    XnUInt64 depth_focal_length_SXGA_mm;   //in mm
    rc = impl_->depth.GetIntProperty ("ZPD", depth_focal_length_SXGA_mm);
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "ZPD failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }

    XnDouble baseline_local;
    rc = impl_->depth.GetRealProperty ("LDDIS", baseline_local);
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "ZPD failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }

    XnUInt64 shadow_value_local;
    rc = impl_->depth.GetIntProperty ("ShadowValue", shadow_value_local);
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "ShadowValue failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }
    shadow_value = (int)shadow_value_local;

    XnUInt64 no_sample_value_local;
    rc = impl_->depth.GetIntProperty ("NoSampleValue", no_sample_value_local);
    if (rc != XN_STATUS_OK)
    {
        sprintf (impl_->strError, "NoSampleValue failed: %s\n", xnGetStatusString (rc));
        REPORT_ERROR (impl_->strError);
    }
    no_sample_value = (int)no_sample_value_local;


    // baseline from cm -> mm
    baseline = (float)(baseline_local * 10);

    //focal length from mm -> pixels (valid for 1280x1024)
    float depth_focal_length_SXGA = static_cast<float>(depth_focal_length_SXGA_mm / pixelSize);
    depth_focal_length_VGA = depth_focal_length_SXGA / 2;
}

bool kfusion::OpenNISource::setRegistration (bool value)
{
	if (fileType == FIFE_PICTURE)
		return 1;
    XnStatus rc = XN_STATUS_OK;

    if (value)
    {
        if (!impl_->has_image)
            return false;

        if (impl_->depth.GetAlternativeViewPointCap ().IsViewPointAs (impl_->image) )
            return true;

        if (!impl_->depth.GetAlternativeViewPointCap ().IsViewPointSupported (impl_->image) )
        {
            printf ("SetRegistration failed: Unsupported viewpoint.\n");
            return false;
        }

        rc = impl_->depth.GetAlternativeViewPointCap ().SetViewPoint (impl_->image);
        if (rc != XN_STATUS_OK)
            printf ("SetRegistration failed: %s\n", xnGetStatusString (rc));

    }
    else   // "off"
    {
        rc = impl_->depth.GetAlternativeViewPointCap ().ResetViewPoint ();
        if (rc != XN_STATUS_OK)
            printf ("SetRegistration failed: %s\n", xnGetStatusString (rc));
    }

    getParams ();
    return rc == XN_STATUS_OK;
}
