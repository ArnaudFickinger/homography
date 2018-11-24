#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <vector>

#include <iostream>

#include "image.h"
#include "homographie.hpp"


using namespace std;
using namespace cv;

int main()
{
	Image<uchar> I1 = Image<uchar>(imread("IMG_0045.JPG", CV_LOAD_IMAGE_GRAYSCALE));
	Image<uchar> I2 = Image<uchar>(imread("IMG_0046.JPG", CV_LOAD_IMAGE_GRAYSCALE));
	
	namedWindow("I1", 1);
	namedWindow("I2", 1);
	imshow("I1", I1);
	imshow("I2", I2);

    Ptr<AKAZE> akaze = AKAZE::create();
    
    vector<KeyPoint> m1, m2;
    
    Mat d1, d2;
    
    akaze->detectAndCompute(I1, noArray(), m1, d1);
    akaze->detectAndCompute(I2, noArray(), m2, d2);
    
	
    
    Mat J;
    drawKeypoints(I1, m1, J);
    imshow("J", J);

    
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(d1, d2, nn_matches, 2);
    
    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    double min = nn_matches[0][0].distance;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        
        if(dist1 < 0.8f * dist2) {
            if(dist1<min){
                min = dist1;
            }
            matched1.push_back(m1[first.queryIdx]);
            matched2.push_back(m2[first.trainIdx]);
            good_matches.push_back(first);
        }
    }
    
    vector<DMatch> good_matches2;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        
        if(dist1 < 10*min) {
            
            good_matches2.push_back(first);
        }
    }
    
    
	
    Mat match;
    drawMatches(I1, m1, I2, m2, good_matches2, match);
    imshow("match", match);
    
	
    vector<Point2f> obj;
    vector<Point2f> scene;
    for( int i = 0; i < good_matches2.size(); i++ )
    {
        obj.push_back( m1[ good_matches2[i].queryIdx ].pt );
        scene.push_back( m2[ good_matches2[i].trainIdx ].pt );
    }
    
    
    Mat H = ransacGeneral(obj, scene);

    
    Mat K(2 * I1.cols, I1.rows, CV_8U);
    Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
    warpAffine(I1, K, trans_mat, Size( 2*I1.cols, I1.rows));
    Mat K2;
    warpPerspective(I2, K2, H, K.size());
    
   
    for (int i = 0; i < 532 ; i++) {
        for (int j = 0; j < 1400; j++) {
            if (K.at<float>(Point(j,i)) == 0){
                K.at<float>(Point(j,i)) = K2.at<float>(Point(j,i));
            }
        }
    }
    
    imshow("I1I2", K) ;
    waitKey(0);
    return 0 ;
}
    

