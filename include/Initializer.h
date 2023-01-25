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
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>
#include "Frame.h"


namespace ORB_SLAM2
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer
{
    typedef pair<int,int> Match;

public:

    // Fix the reference frame
    Initializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);
    

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);
    bool InitializeWAF(const Frame &CurrentFrame, const vector<int> &vMatches12, const vector<int> &vMatches12_middle, const vector<int> &vMatches12_high,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<cv::Point3f> &vP3D_middle, vector<cv::Point3f> &vP3D_high, vector<bool> &vbTriangulated, vector<bool> &vbTriangulated_middle, vector<bool> &vbTriangulated_high);

private:

    void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
    void FindHomographyWAF(vector<bool> &vbMatchesInliers, vector<bool> &vbMatchesInliers_middle, vector<bool> &vbMatchesInliers_high, float &score, cv::Mat &H21);
 
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);
    void FindFundamentalWAF(vector<bool> &vbMatchesInliers,vector<bool> &vbMatchesInliers_mdddle,vector<bool> &vbMatchesInliers_high, float &score, cv::Mat &F21);


    cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);
    float CheckHomographyMiddle(const cv::Mat &H21_middle, const cv::Mat &H12_middle, vector<bool> &vbMatchesInliers_middle, float sigma);
    float CheckHomographyHigh(const cv::Mat &H21_high, const cv::Mat &H12_high, vector<bool> &vbMatchesInliers_high, float sigma);

    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);
    float CheckFundamentalMiddle(const cv::Mat &F21_middle, vector<bool> &vbMatchesInliers_middle, float sigma);
    float CheckFundamentalHigh(const cv::Mat &F21_high, vector<bool> &vbMatchesInliers_high, float sigma);

    

    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    bool ReconstructF_WAF(vector<bool> &vbMatchesInliers, vector<bool> &vbMatchesInliers_middle, vector<bool> &vbMatchesInliers_high, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<cv::Point3f> &vP3D_middle, vector<cv::Point3f> &vP3D_high, vector<bool> &vbTriangulated, vector<bool> &vbTriangulated_middle, vector<bool> &vbTriangulated_high, float minParallax, int minTriangulated);


    bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
    bool ReconstructH_WAF(vector<bool> &vbMatchesInliers, vector<bool> &vbMatchesInliers_middle, vector<bool> &vbMatchesInliers_high, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<cv::Point3f> &vP3D_middle, vector<cv::Point3f> &vP3D_high, vector<bool> &vbTriangulated, vector<bool> &vbTriangulated_middle, vector<bool> &vbTriangulated_high, float minParallax, int minTriangulated);


    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


    // Keypoints from Reference Frame (Frame 1)
    vector<cv::KeyPoint> mvKeys1, mvKeys1_middle, mvKeys1_high;

    // Keypoints from Current Frame (Frame 2)
    vector<cv::KeyPoint> mvKeys2, mvKeys2_middle, mvKeys2_high;

    // Current Matches from Reference to Current
    vector<Match> mvMatches12, mvMatches12_middle, mvMatches12_high;
    vector<bool> mvbMatched1, mvbMatched1_middle, mvbMatched1_high;

    // Calibration
    cv::Mat mK;

    // Standard Deviation and Variance
    float mSigma, mSigma2;

    // Ransac max iterations
    int mMaxIterations;

    // Ransac sets
    vector<vector<size_t> > mvSets, mvSets_middle, mvSets_high;   

};

} //namespace ORB_SLAM

#endif // INITIALIZER_H
