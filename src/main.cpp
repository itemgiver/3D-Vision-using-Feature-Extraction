#include <stdio.h>
#include <iostream>
#include <time.h>
#include <algorithm>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define uchar unsigned char

using namespace std;
using namespace cv;
using namespace xfeatures2d;

extern CvScalar getcolor(float scale);
extern void drawBlendedMatches(InputArray img1, const std::vector<KeyPoint>& keypoints1,
	InputArray img2, const std::vector<KeyPoint>& keypoints2,
	const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
	const Scalar& singlePointColor, const std::vector<char>& matchesMask,
	int flags, float offset, float diff, int mode);
extern void drawBlendedMatchesWhite(InputArray img1, const std::vector<KeyPoint>& keypoints1,
	InputArray img2, const std::vector<KeyPoint>& keypoints2,
	const std::vector<DMatch>& matches1to2, InputOutputArray outImg,
	const Scalar& singlePointColor, const std::vector<char>& matchesMask,
	int flags, int mode);

char f1[] = "img\\ff1.jpg";
char f2[] = "img\\ff2.jpg";
char f3[] = "img\\ff3.jpg";
char savestr[] = "img\\r .jpg";
char textstr[] = "img\\txtfile .txt";

int main()
{
	FILE* l = fopen("img\\log.txt", "at");
	fprintf(l, "%s + %s\n", f1, f2);
	clock_t begin;
	Ptr<Feature2D> f2d = SURF::create(70,4,3,false,true);
	Mat img_1_, img_2_, img_1, img_2,img_3_,img_3;
	int divide = 8;
	int runtime = 7;
	int goodmatchDivider = 2;
	bool allmatchoutput = true;
	bool txtoutput = true;
	bool imgoutput = true;
	bool keypointoutput = true;

	begin = clock();
	img_1_ = imread(f1);
	cv::resize(img_1_, img_1, Size(img_1_.cols / divide, img_1_.rows / divide));
	img_2_ = imread(f2);
	cv::resize(img_2_, img_2, Size(img_2_.cols / divide, img_2_.rows / divide));
	img_3_ = imread(f3);
	cv::resize(img_3_,img_3,Size(img_3_.cols/divide,img_3_.rows/divide));

	int x, y, z;
	x = img_1.cols; y = img_1.rows; z = img_1.channels();
	printf("(%d, %d) %dchannels\n", x, y, z);
	fprintf(l, "(%d, %d) %dchannels\n", x, y, z);
	fprintf(l, "runtime: %d goodmatchDiv: %d\n\n", runtime, goodmatchDivider);
	printf("imread: %.3f\n", (clock() - begin) / (float)CLOCKS_PER_SEC);
	fprintf(l, "imread: %.3f\n", (clock() - begin) / (float)CLOCKS_PER_SEC);

	//-- Step 1: Detect the keypoints:
	vector<KeyPoint> keypoints_1, keypoints_2,keypoints_3;
	begin = clock();
	f2d->detect(img_1, keypoints_1);
	printf("img1detected: %d  time: %.3f\n", keypoints_1.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);
	fprintf(l, "img1detected: %d  time: %.3f\n", keypoints_1.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);
	begin = clock();
	f2d->detect(img_2, keypoints_2);
	printf("img2detected: %d  time: %.3f\n", keypoints_2.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);
	fprintf(l, "img2detected: %d  time: %.3f\n", keypoints_2.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);
	begin = clock();
	f2d->detect(img_3,keypoints_3);
	printf("img3detected: %d  time: %.3f\n", keypoints_3.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);
	fprintf(l, "img3detected: %d  time: %.3f\n", keypoints_3.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);

	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2,descriptors_3;
	begin = clock();
	f2d->compute(img_1, keypoints_1, descriptors_1);
	printf("rows: %d, columns: %d\n", descriptors_1.rows, descriptors_1.cols);
	printf("img1computed  time: %.3f\n", (clock() - begin) / (float)CLOCKS_PER_SEC);
	fprintf(l, "img1computed  time: %.3f\n", (clock() - begin) / (float)CLOCKS_PER_SEC);
	begin = clock();
	f2d->compute(img_2, keypoints_2, descriptors_2);
	printf("img2computed  time: %.3f\n", (clock() - begin) / (float)CLOCKS_PER_SEC);
	fprintf(l, "img2computed  time: %.3f\n", (clock() - begin) / (float)CLOCKS_PER_SEC);
	begin = clock();
	f2d->compute(img_3,keypoints_3,descriptors_3);
	printf("img3computed  time: %.3f\n", (clock() - begin) / (float)CLOCKS_PER_SEC);
	fprintf(l, "img3computed  time: %.3f\n", (clock() - begin) / (float)CLOCKS_PER_SEC);

	Mat op;
	drawKeypoints(img_1, keypoints_1, op, getcolor(1));
	if (keypointoutput)imwrite("img\\keypoints1.jpg", op);
	drawKeypoints(img_2, keypoints_2, op, getcolor(0.5f));
	if (keypointoutput)imwrite("img\\keypoints2.jpg", op);
	drawKeypoints(img_1, keypoints_1, op, getcolor(0));
	if (keypointoutput)imwrite("img\\keypoints1.jpg", op);

	//-- Step 3: Matching descriptor vectors using BFMatcher :
	FlannBasedMatcher matcher;

	begin = clock();
	vector< DMatch > matches1, goodmatches1,matches2,goodmatches2,matches3,goodmatches3;
	matcher.match(descriptors_1, descriptors_2, matches1);
	matcher.match(descriptors_2,descriptors_3,matches2);
	matcher.match(descriptors_3,descriptors_1,matches3);

	//printf("matches: %d  time: %.3f\n", matches.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);
	//fprintf(l, "matches: %d  time: %.3f\n", matches.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);

	//-- Step 4: Filtering goodmatches
	sort(matches1.begin(), matches1.end(), [&](DMatch cmpx, DMatch cmpy) {
		return cmpx.distance < cmpy.distance;
	});
	sort(matches2.begin(), matches2.end(), [&](DMatch cmpx, DMatch cmpy) {
		return cmpx.distance < cmpy.distance;
	});
	sort(matches3.begin(), matches3.end(), [&](DMatch cmpx, DMatch cmpy) {
		return cmpx.distance < cmpy.distance;
	});
	vector<DMatch> tmpmatches1,tmpmatches2,tmpmatches3,tmpmatches;
	for (int i = 0; i < (matches1.size() / goodmatchDivider); i++) tmpmatches1.push_back(matches1[i]);
	for(int i=0; i<(matches2.size()/goodmatchDivider); i++) tmpmatches2.push_back(matches2[i]);
	for(int i=0; i<(matches3.size()/goodmatchDivider); i++) tmpmatches3.push_back(matches3[i]);

	/*Mat output;
	if (allmatchoutput) {
		matches = tmpmatches;
		float distmin = 10000000000000, distmax = 0;
		for (int i = 0; i < matches.size(); i++) {
			Point2f p1 = keypoints_1[matches[i].queryIdx].pt;
			Point2f p2 = keypoints_2[matches[i].trainIdx].pt;
			int x1 = p1.x;
			int x2 = p2.x;
			int y1 = p1.y;
			int y2 = p2.y;
			float d = sqrtf((float)(x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
			if (d < distmin) distmin = d;
			if (d > distmax) distmax = d;
		}
		drawBlendedMatches(img_1, keypoints_1, img_2, keypoints_2, matches, output, Scalar(255, 255, 255, 100), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS, distmin, distmax - distmin, 0);
		imwrite("img\\allMatches.jpg", output);
	}*/


	begin = clock();
	int *query1, *train1;
	query1 = (int*)malloc(sizeof(int)*(max(keypoints_1.size(),keypoints_2.size())+10));
	train1 = (int*)malloc(sizeof(int)*(max(keypoints_1.size(), keypoints_2.size()) + 10));
	for (int i = 0; i < max(keypoints_1.size(), keypoints_2.size()) + 10; i++) {
		query1[i] = train1[i] = 0;
	}
	int *query2, *train2;
	query2 = (int*)malloc(sizeof(int)*(max(keypoints_2.size(),keypoints_3.size())+10));
	train2 = (int*)malloc(sizeof(int)*(max(keypoints_2.size(), keypoints_3.size()) + 10));
	for (int i = 0; i < max(keypoints_2.size(), keypoints_3.size()) + 10; i++) {
		query2[i] = train2[i] = 0;
	}
	int *query3, *train3;
	query3 = (int*)malloc(sizeof(int)*(max(keypoints_3.size(),keypoints_1.size())+10));
	train3 = (int*)malloc(sizeof(int)*(max(keypoints_3.size(), keypoints_1.size()) + 10));
	for (int i = 0; i < max(keypoints_3.size(), keypoints_1.size()) + 10; i++) {
		query3[i] = train3[i] = 0;
	}
	//int j = 0;
	
	while (runtime--) {
			//j++;
			printf("%d %d %d\n", tmpmatches1.size(), tmpmatches2.size(), tmpmatches3.size());
			vector<Point2f> tmp_obj;
			vector<Point2f> tmp_scene;
			{
			for (int i = 0; i < tmpmatches1.size(); i++) {
				//-- Get the keypoints from the good matches
				tmp_obj.push_back(keypoints_1[tmpmatches1[i].queryIdx].pt);
				tmp_scene.push_back(keypoints_2[tmpmatches1[i].trainIdx].pt);
			}
			Mat tmpMask;
			if (tmpmatches1.size() > 0)
			{
				Mat tmpH = findHomography(tmp_obj, tmp_scene, CV_RANSAC, 2.0, tmpMask);

				for (int i = 0; i < tmpmatches1.size(); i++) {
					if (tmpMask.at<uchar>(i) != 0)  // RANSAC selection
					{
						goodmatches1.push_back(tmpmatches1[i]);
						query1[tmpmatches1[i].queryIdx]++;
						train1[tmpmatches1[i].trainIdx]++;
						//printf("%d %d\n", tmpmatches[i].queryIdx, query[tmpmatches[i].queryIdx]);
					}
					else {
						if (query1[tmpmatches1[i].queryIdx] >= 1 || train1[tmpmatches1[i].trainIdx] >= 1) continue;
						tmpmatches.push_back(tmpmatches1[i]);
					}
				}
				tmpmatches1.clear();
				for (int i = 0; i < tmpmatches.size(); i++) tmpmatches1.push_back(tmpmatches[i]);
				tmpmatches.clear();
			}
		}
		//--------------
		{
			tmp_obj.clear(); tmp_scene.clear();

			for (int i = 0; i < tmpmatches2.size(); i++) {
				//-- Get the keypoints from the good matches
				tmp_obj.push_back(keypoints_2[tmpmatches2[i].queryIdx].pt);
				tmp_scene.push_back(keypoints_3[tmpmatches2[i].trainIdx].pt);
			}

			Mat tmpMask;
			if (tmpmatches2.size() > 0)
			{
				Mat tmpH = findHomography(tmp_obj, tmp_scene, CV_RANSAC, 2.0, tmpMask);

				for (int i = 0; i < tmpmatches2.size(); i++) {
					if (tmpMask.at<uchar>(i) != 0)  // RANSAC selection
					{
						goodmatches2.push_back(tmpmatches2[i]);
						query2[tmpmatches2[i].queryIdx]++;
						train2[tmpmatches2[i].trainIdx]++;
						//printf("%d %d\n", tmpmatches[i].queryIdx, query[tmpmatches[i].queryIdx]);
					}
					else {
						if (query2[tmpmatches2[i].queryIdx] >= 1 || train2[tmpmatches2[i].trainIdx] >= 1) continue;
						tmpmatches.push_back(tmpmatches2[i]);
					}
				}
				tmpmatches2.clear();
				for (int i = 0; i < tmpmatches.size(); i++) tmpmatches2.push_back(tmpmatches[i]);
				tmpmatches.clear();
			}
		}
		//----------------
		{
			tmp_obj.clear(); tmp_scene.clear();
			for (int i = 0; i < tmpmatches3.size(); i++) {
				//-- Get the keypoints from the good matches
				tmp_obj.push_back(keypoints_3[tmpmatches3[i].queryIdx].pt);
				tmp_scene.push_back(keypoints_1[tmpmatches3[i].trainIdx].pt);
			}
			Mat tmpMask;
			if (tmpmatches3.size() > 0)
			{
				Mat tmpH = findHomography(tmp_obj, tmp_scene, CV_RANSAC, 2.0, tmpMask);

				for (int i = 0; i < tmpmatches3.size(); i++) {
					if (tmpMask.at<uchar>(i) != 0)  // RANSAC selection
					{
						goodmatches3.push_back(tmpmatches3[i]);
						query3[tmpmatches3[i].queryIdx]++;
						train3[tmpmatches3[i].trainIdx]++;
						//printf("%d %d\n", tmpmatches[i].queryIdx, query[tmpmatches[i].queryIdx]);
					}
					else {
						if (query3[tmpmatches3[i].queryIdx] >= 1 || train3[tmpmatches3[i].trainIdx] >= 1) continue;
						tmpmatches.push_back(tmpmatches3[i]);
					}
				}
				tmpmatches3.clear();
				for (int i = 0; i < tmpmatches.size(); i++) tmpmatches3.push_back(tmpmatches[i]);
				tmpmatches.clear();
			}
		}
		/*printf("ransac %d  matches: %d  time: %.3f\n", j, goodmatches.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);
		fprintf(l, "ransac %d  matches: %d  time: %.3f\n", j, goodmatches.size(), (clock() - begin) / (float)CLOCKS_PER_SEC);

		float distmin = 10000000000000, distmax = 0;
		textstr[11] = j + '1' - 1;
		FILE* f = fopen(textstr, "wt");
		for (int i = 0; i < goodmatches.size(); i++) {
			Point2f p1 = keypoints_1[goodmatches[i].queryIdx].pt;
			Point2f p2 = keypoints_2[goodmatches[i].trainIdx].pt;
			int x1 = p1.x;
			int x2 = p2.x;
			int y1 = p1.y;
			int y2 = p2.y;
			if (txtoutput) fprintf(f, "%d %d %d %d\n", x1, y1, x2, y2);
			float d = sqrtf((float)(x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
			if (d < distmin) distmin = d;
			if (d > distmax) distmax = d;
		}
		fclose(f);

		savestr[5] = j + '1' - 1;*/
	}
	//drawBlendedMatches(img_1, keypoints_1, img_2, keypoints_2, goodmatches, output, Scalar(255, 255, 255, 100), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS, distmin, distmax - distmin, 0);
	//if (imgoutput) imwrite(savestr, output);
	vector<DMatch> finmatches;
	FILE* f = fopen(textstr, "wt");
	for(int i=0; i<goodmatches1.size(); i++){
		for(int j=0; j<goodmatches2.size(); j++){
			if(goodmatches1[i].trainIdx != goodmatches2[j].queryIdx) continue;
			for(int k=0; k<goodmatches3.size(); k++){
				if(goodmatches2[j].trainIdx != goodmatches3[k].queryIdx) continue;
				if(goodmatches3[k].trainIdx != goodmatches1[i].queryIdx) continue;
				//이러면 찾은 거임 ㅇㅋ
				//image 1에서 keypoint1[goodmtches1[i].queryIdx]
				//image2에서 keypoin[goodmatches2[i].queryIdx];
				Point2f p1 = keypoints_1[goodmatches1[i].queryIdx].pt;
				Point2f p2 = keypoints_2[goodmatches2[j].queryIdx].pt;
				Point2f p3 = keypoints_3[goodmatches3[k].queryIdx].pt;
				int x1 = p1.x;
				int x2 = p2.x;
				int y1 = p1.y;
				int y2 = p2.y;
				int x3 = p3.x;
				int y3 = p3.y;
				if (txtoutput) fprintf(f, "%d %d %d %d %d %d\n", x1, y1, x2, y2, x3, y3);
				finmatches.push_back(goodmatches1[i]);
			}
		}
	}
	fclose(f);
	Mat output;
	drawBlendedMatchesWhite(img_1, keypoints_1, img_2, keypoints_2, finmatches, output, Scalar(255, 255, 255, 100), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | DrawMatchesFlags::DRAW_RICH_KEYPOINTS, 0);
	imwrite("img\\allMatches.jpg", output);


	fprintf(l, "=================================================\n\n");
	fclose(l);
	return 0;
}
