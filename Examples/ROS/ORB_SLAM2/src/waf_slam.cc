/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include<opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include"../../../include/System.h"

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM);
    void GrabImage(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageWAF(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat ReadMatFromTxt(std::string filename, int rows,int cols);
    void setImagePointer(cv_bridge::CvImageConstPtr cv_ptr, cv::Mat Syn_x, cv::Mat Syn_y, cv::Mat &dst);
    ORB_SLAM2::System* mpSLAM;
private:
    cv::Mat Syn_x_low, Syn_x_middle, Syn_x_high;
    cv::Mat Syn_y_low, Syn_y_middle, Syn_y_high;
    cv::Mat img_low, img_middle, img_high;
};

ImageGrabber::ImageGrabber(ORB_SLAM2::System* pSLAM):mpSLAM(pSLAM)
{
    Syn_x_low = ReadMatFromTxt("Syn_x_t.txt", 168,425);
    Syn_y_low = ReadMatFromTxt("Syn_y_t.txt", 168,425);
    Syn_x_middle = ReadMatFromTxt("Syn_x_t.txt", 168,425);
    Syn_y_middle = ReadMatFromTxt("Syn_y_t.txt", 168,425);
    Syn_x_high = ReadMatFromTxt("Syn_x_t.txt", 168,425);
    Syn_y_high = ReadMatFromTxt("Syn_y_t.txt", 168,425);

}


void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    setImagePointer(cv_ptr, Syn_x_low, Syn_y_low, img_low);
    cv::waitKey(1);
    mpSLAM->TrackMonocular(img_low,cv_ptr->header.stamp.toSec());
}


void ImageGrabber::GrabImageWAF(const sensor_msgs::ImageConstPtr& msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    setImagePointer(cv_ptr, Syn_x_low, Syn_y_low, img_low);
    setImagePointer(cv_ptr, Syn_x_middle, Syn_y_middle, img_middle);
    setImagePointer(cv_ptr, Syn_x_high, Syn_y_high, img_high);
    //mpSLAM->TrackMonocular(img_low, cv_ptr->header.stamp.toSec());
    cv::waitKey(1);
    mpSLAM->TrackMonocularWAF(img_low, img_middle, img_high, cv_ptr->header.stamp.toSec());
}

cv::Mat ImageGrabber::ReadMatFromTxt(std::string filename, int rows,int cols)
{
    double m;
    cv::Mat out = cv::Mat::zeros(rows, cols, CV_64FC1);//Matrix to store values

    std::ifstream fileStream(filename);
    if (!fileStream)
    {
        std::cout << "FILE NOT FOUND" << std::endl;
        std::cin.get();
        exit(1);
    }
    int cnt = 0;//index starts from 0
    while (fileStream >> m)
    {
        int temprow = cnt / cols;
        int tempcol = cnt % cols;
        out.at<double>(temprow, tempcol) = m;
        cnt++;
    }
    return out;
}

void ImageGrabber::setImagePointer(cv_bridge::CvImageConstPtr cv_ptr, cv::Mat Syn_x, cv::Mat Syn_y, cv::Mat &dst)
{
  int width = Syn_x.cols;
  int height = Syn_y.rows;
  dst = cv::Mat::zeros(height, width, CV_8UC3);
  int channels = dst.channels();
  int step = dst.step;
  int elemsize = dst.elemSize();
  for(int j=0; j<height; j++)
  {
    for(int i=0; i<width; i++)
    {
      int x = Syn_x.at<double>(j,i);
      int y = Syn_y.at<double>(j,i);
      memcpy(&dst.data[j*step + i*elemsize], &cv_ptr->image.data[y*cv_ptr->image.step + x*cv_ptr->image.elemSize()], elemsize);
    }
  }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "WAF");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM2 Mono path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }    
    bool waf = true;
    ros::NodeHandle nodeHandler;
    ros::Subscriber sub;

    // Create SLAM system. It initializes all system threads and gets ready to process frames. 
    //ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true,true);
    
    if(waf){
        ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true,true);
        ImageGrabber igb(&SLAM);
        sub = nodeHandler.subscribe("/camera/waf_capture", 1, &ImageGrabber::GrabImageWAF,&igb);
        ros::spin();
        // Stop all threads
        SLAM.Shutdown();
        // Save camera trajectory
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
        ros::shutdown();
    }
    else{
        ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);
        ImageGrabber igb(&SLAM);
        sub = nodeHandler.subscribe("/camera/waf_capture", 1, &ImageGrabber::GrabImage,&igb);
        ros::spin();
        // Stop all threads
        SLAM.Shutdown();
        // Save camera trajectory
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
        ros::shutdown();
    }
    

    return 0;
}
