#pragma once
//#include <windows.h>
//#include "NuiApi.h"
#include <kfusion/kinfu.hpp>
#include <opencv2/core/core.hpp>
#include <string>
enum TYPE
{
	FIFE_PICTURE,
	FIFE_ONI
};
namespace kfusion
{
    class KF_EXPORTS OpenNISource
    {
    public:
        typedef kfusion::PixelRGB RGB24;

        enum { PROP_OPENNI_REGISTRATION_ON  = 104 };

        OpenNISource();
        OpenNISource(int device);
		OpenNISource(const std::string& oni_filename, bool repeat = false, TYPE type=FIFE_ONI);

		void open(int device);
		void open(const std::string& oni_filename, bool repeat = false, TYPE type = FIFE_ONI);
        void release();

        ~OpenNISource();

        bool grab(cv::Mat &depth, cv::Mat &image);

        //parameters taken from camera/oni
        int shadow_value, no_sample_value;
        float depth_focal_length_VGA;
        float baseline;               // mm
        double pixelSize;             // mm
        unsigned short max_depth;     // mm

		TYPE fileType;
		int m_current_frame;
        bool setRegistration (bool value = false);
    private:
        struct Impl;
        cv::Ptr<Impl> impl_;
        void getParams ();
	/*	INuiSensor*                         m_pNuiSensor;

		HANDLE                              m_hNextDepthFrameEvent;
		HANDLE                              m_pDepthStreamHandle;
		HANDLE                              m_hNextColorFrameEvent;
		HANDLE                              m_pColorStreamHandle;
		bool                                m_bNearMode;

		static const NUI_IMAGE_RESOLUTION   cDepthResolution = NUI_IMAGE_RESOLUTION_640x480;
		static const NUI_IMAGE_RESOLUTION   cColorResolution = NUI_IMAGE_RESOLUTION_640x480;
		bool                                m_bDepthReceived;
		bool                                m_bColorReceived;*/
    };
}
