#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <random>
#include <iterator>
#include <math.h>
#include <time.h>
// #include <algorithm>
// #include <vector>
#include "image.h"

#define w 400

using namespace std;
using namespace cv;

void computeA(Mat &A, vector<Point2f> &obj, vector<Point2f> &scene, int a, int b, int c, int d){
    A = (Mat_<float>(9,9) << 0, 0, 0, -obj[a].x, -obj[a].y, -1, scene[a].y*obj[a].x, scene[a].y*obj[a].y, scene[a].y,
         obj[a].x, obj[a].y, 1, 0, 0, 0, -obj[a].x*scene[a].x, -scene[a].x*obj[a].y, -scene[a].x,
         0, 0, 0, -obj[b].x, -obj[b].y, -1, scene[b].y*obj[b].x, scene[b].y*obj[b].y, scene[b].y,
         obj[b].x, obj[b].y, 1, 0, 0, 0, -obj[b].x*scene[b].x, -scene[b].x*obj[b].y, -scene[b].x,
         0, 0, 0, -obj[c].x, -obj[c].y, -1, scene[c].y*obj[c].x, scene[c].y*obj[c].y, scene[b].y,
         obj[c].x, obj[c].y, 1, 0, 0, 0, -obj[c].x*scene[c].x, -scene[c].x*obj[c].y, -scene[c].x,
         0, 0, 0, -obj[d].x, -obj[d].y, -1, scene[d].y*obj[d].x, scene[d].y*obj[d].y, scene[d].y,
         obj[d].x, obj[d].y, 1, 0, 0, 0, -obj[d].x*scene[d].x, -scene[d].x*obj[d].y, -scene[d].x,
         0, 0, 0, 0, 0, 0, 0, 0, 1);
}

Mat ransacGeneral(vector<Point2f> &obj, vector<Point2f> &scene) {
    int iter = 1000; // Reset this to 100 later
    float threshDist = 3.0; // Threshold of distances between points and line
    float ransacRatio = 0.50; // Threshold of number of inliers to assert model fits data well
    float numSamples = (float)obj.size();
    float bestRatio = 0; // Best ratio
    float num;
    
    Point2f p1, p2;
    
    Mat_<float> A, H, Z, bestH, s, u, vt, v;
    Z = Mat::zeros(1,9, CV_32F);
    Z.at<float>(8)=1;
    cout << Z << endl;
    
    int sz = obj.size();
    
    int a,b,c,d;
    vector<int> inliers;
    
    vector<float> objPoint;
    vector<float> scnPoint;
    
    for (int i = 0; i < iter; i++) {
        a = rand() % sz;
        b = rand() % sz;
        while (b == a){
            b = rand() % sz;
        }
        c = rand() % sz;
        while (c == a || c == b){
            c = rand() % sz;
        }
        d = rand() % sz;
        while (d == a || d == b || d == c){
            d = rand() % sz;
        }
        computeA(A, obj, scene, a, b, c, d); // AH = 0
        
        Z = Z.t();
        solve(A, Z, H, DECOMP_SVD);
        if(H.size().width!=1) continue;
        H = H.reshape(0, 3);
        
        cout << "[ INFO ] Iteration " << i << endl;
        
        num = 0;
        Point2f iP;
        float dist;
        for (int i = 0; i<obj.size(); i++) {
            if(i == a || i == b || i == c || i == d){
                continue;
            }
            objPoint.clear();
            scnPoint.clear();
            
            objPoint.push_back(obj[i].x);
            objPoint.push_back(obj[i].y);
            objPoint.push_back(1);
            
            scnPoint.push_back(scene[i].x);
            scnPoint.push_back(scene[i].y);
            scnPoint.push_back(1);
        
            double dist = norm(Mat_<float>(scnPoint), H * Mat_<float>(objPoint));
            cout << "dist " << dist << endl;
            
            if (dist < threshDist) {
                inliers.push_back(i);
                num++;
            }
            
        }
        
        cout << "num/numSamples " << num/numSamples << endl;
        if (num/numSamples > bestRatio) {
            bestRatio = num/numSamples;
            bestH = H;
        }
        
        if (num > numSamples * ransacRatio) {
            cout << "Found good enough model" << endl;
            break;
        }
    }
    
    cout << bestH << endl;
    return bestH;
}

// Correlation
double mean(const Image<float>& I,Point m,int n) {
    double s=0;
    for (int j=-n;j<=n;j++)
        for (int i=-n;i<=n;i++)
            s+=I(m+Point(i,j));
    return s/(2*n+1)/(2*n+1);
}

double corr(const Image<float>& I1,Point m1,const Image<float>& I2,Point m2,int n) {
    double M1=mean(I1,m1,n);
    double M2=mean(I2,m2,n);
    double rho=0;
    for (int j=-n;j<=n;j++)
        for (int i=-n;i<=n;i++) {
            rho+=(I1(m1+Point(i,j))-M1)*(I2(m2+Point(i,j))-M2);
        }
    return rho;
}

double NCC(const Image<float>& I1,Point m1,const Image<float>& I2,Point m2,int n) {
    if (m1.x<n || m1.x>=I1.width()-n || m1.y<n || m1.y>=I1.height()-n) return -1;
    if (m2.x<n || m2.x>=I2.width()-n || m2.y<n || m2.y>=I2.height()-n) return -1;
    double c1=corr(I1,m1,I1,m1,n);
    if (c1==0) return -1;
    double c2=corr(I2,m2,I2,m2,n);
    if (c2==0) return -1;
    return corr(I1,m1,I2,m2,n)/sqrt(c1*c2);
}
