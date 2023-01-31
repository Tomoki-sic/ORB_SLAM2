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

#ifndef FRAMEDRAWER_H
#define FRAMEDRAWER_H

#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include<mutex>


namespace ORB_SLAM2
{

class Tracking;
class Viewer;

class FrameDrawer
{
public:
    FrameDrawer(Map* pMap);
    FrameDrawer(Map* pMap, Map* pMap_middle, Map* pMap_high);
    

    // Update info from the last processed frame.
    void Update(Tracking *pTracker);
    void UpdateWAF(Tracking *pTracker);

    // Draw last processed frame.
    cv::Mat DrawFrame();
    cv::Mat DrawFrameMiddle();
    cv::Mat DrawFrameHigh();

protected:
    void DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText);
    // Info of the frame to be drawn
    cv::Mat mIm, mIm_middle, mIm_high;
    int N, N_middle, N_high;
    vector<cv::KeyPoint> mvCurrentKeys, mvCurrentKeys_middle, mvCurrentKeys_high;
    vector<bool> mvbMap, mvbMap_midle, mvbMap_high, mvbVO, mvbVO_middle, mvbVO_high;
    bool mbOnlyTracking;
    int mnTracked, mnTrackedVO;
    vector<cv::KeyPoint> mvIniKeys, mvIniKeys_middle, mvIniKeys_high;
    vector<int> mvIniMatches, mvIniMatches_middle, mvIniMatches_high;
    int mState;

    Map* mpMap, *mpMap_middle, *mpMap_high;

    std::mutex mMutex;
};

} //namespace ORB_SLAM

#endif // FRAMEDRAWER_H
