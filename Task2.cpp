#include <stdio.h>
#include <opencv2/opencv.hpp>        
#include <opencv2/core/core.hpp>    
#include <opencv2/highgui/highgui_c.h>
#include "opencv2/objdetect/objdetect_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>
#include <algorithm>
#include <string>

using namespace cv;
using namespace std;

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;

int mod(int a, int b) {
	int x = a % b;
	if (x < 0) {
		return b + a;
	}
	else return x;
}

int ***malloc3dArray(int dim1, int dim2, int dim3)
{
	int i, j, k;

	int ***array = (int ***)malloc(dim1 * sizeof(int **));

	for (i = 0; i < dim1; i++) {

		array[i] = (int **)malloc(dim2 * sizeof(int *));

		for (j = 0; j < dim2; j++) {

			array[i][j] = (int *)malloc(dim3 * sizeof(int));
		}

	}
	return array;

}

tuple<int, int> findCentreOfBox(Rect box) {
	int a = (box.x+box.width) - box.x;
	int b = (box.y+box.height) - box.y;
	int da = a / 2;
	int db = b / 2;

	tuple<int, int> centre = make_tuple(box.x + da, box.y + db);
	
	return centre;
}

/** @function calculatef1Score */
float f1Score(vector<tuple<int, int> > groundTruth, vector<Rect> detectedDartboards, int detectionThreshold) {
	int trueFaces = (int)groundTruth.size();
	int identifiedFaces = (int)detectedDartboards.size();

	cout << "True darboards: " << trueFaces << "\n";
	cout << "Identified dartboards: " << identifiedFaces << "\n";
	int truePositive = 0;

	for (int i = 0; i < groundTruth.size(); i++) {
		for (int j = 0; j < detectedDartboards.size(); j++) {
			Rect currentRectangle = detectedDartboards[j];
			tuple<int, int> currentTruth = groundTruth[i];
			tuple<int, int>  rectCentre = findCentreOfBox(currentRectangle);
			if ((abs(get<0>(rectCentre) - get<0>(currentTruth)) < (detectionThreshold)) && (abs(get<1>(rectCentre) - get<1>(currentTruth)) < (detectionThreshold))) {
				truePositive++;
			}
		}
	}

	cout << "True positive: " << truePositive << "\n";
	if (truePositive == 0) {
		return 0;
	}
	int falsePositive = identifiedFaces - truePositive;
	int falseNegative = ((trueFaces - truePositive) < 0) ? 0 : trueFaces - truePositive;

	cout << "False positive: " << falsePositive << "\n";
	cout << "False negative: " << falseNegative << "\n";
	float precision = (float)truePositive / (truePositive + falsePositive);
	float recall = (float)truePositive / (truePositive + falseNegative);

	cout << "Precision: " << recall << "\n";
	cout << "Recall: " << precision << "\n";
	float f1Score = (precision*recall) / (precision + recall);

	return f1Score*2;
}

vector<tuple<int, int> > findGroundTruth(int name) {

	vector <tuple<int, int>> truth;

	switch (name) {
		case 0:
			truth.clear();
			truth.push_back( make_tuple( 515, 100 ));
			return truth;
		case 1:
			truth.clear();
			truth.push_back( make_tuple( 295, 225 ));
			return truth;
		case 2:
			truth.clear();
			truth.push_back( make_tuple( 148, 140 ));
			return truth;
		case 3:
			truth.clear();
			truth.push_back( make_tuple( 358, 153 ));
			return truth;
		case 4:
			truth.clear();
			truth.push_back( make_tuple( 288, 196 ));
			return truth;
		case 5:
			truth.clear();
			truth.push_back( make_tuple( 487, 194 ));
			return truth;
		case 6:
			truth.clear();
			truth.push_back( make_tuple( 240, 147 ));
			return truth;
		case 7:
			truth.clear();
			truth.push_back( make_tuple( 326, 243 ));
			return truth;
		case 8:
			truth.clear();
			truth.push_back( make_tuple( 900, 277 ));
			truth.push_back( make_tuple( 96, 295 ));
			return truth;
		case 9:
			truth.clear();
			truth.push_back( make_tuple( 319, 162 ));
			return truth;
		case 10:
			truth.clear();
			truth.push_back(make_tuple( 140, 158 ));
			truth.push_back(make_tuple( 611, 169 ));
			truth.push_back(make_tuple( 935, 180 ));
			return truth;
		case 11:
			truth.clear();
			truth.push_back(make_tuple( 204, 143 ));
			return truth;
		case 12:
			truth.clear();
			truth.push_back(make_tuple( 188, 146 ));
			return truth;
		case 13:
			truth.clear();
			truth.push_back(make_tuple( 337, 184 ));
			return truth;
		case 14:
			truth.clear();
			truth.push_back(make_tuple( 184, 163 ));
			truth.push_back(make_tuple( 1050, 156 ));
			return truth;
		case 15:
			truth.clear();
			truth.push_back(make_tuple( 223, 124 ));
			return truth;

	}
	return truth;
}

vector<tuple<int, int> > houghSpace(Mat magnitude, Mat gradient, int threshold, int minRad, int maxRad, string imageName, int detectionThreshold) {

	int ***parameterSpace;
	vector<tuple<int, int>> bestCircles;
	int dim1 = magnitude.rows, dim2 = magnitude.cols;
	int i, j, k;

	parameterSpace = malloc3dArray(dim1, dim2, maxRad);

	for (i = 0; i < dim1; ++i)
		for (j = 0; j < dim2; ++j)
			for (k = minRad; k < maxRad; ++k)
				parameterSpace[i][j][k] = 0;

	for (int x = 0; x < magnitude.rows; x++) {
		for (int y = 0; y < magnitude.cols; y++) {
			if (magnitude.at<uchar>(x, y) == threshold) {
				for (int r = minRad; r < maxRad; r++) {
					float angle = (float) gradient.at<float>(x, y);
					for (int spread = -5; spread <= 5; spread++) {
						float spreadRad = spread * (CV_PI / 180);
						float ang = angle + spreadRad;
						int x_0 = (int) x + r*sin(ang);
						int y_0 = (int) y + r*cos(ang);

						if ((x_0 >= 0 && x_0 < magnitude.rows) && (y_0 >= 0 && y_0 < magnitude.cols)) {
							parameterSpace[x_0][y_0][r] += 1;
						}

						int x_01 = (int) x - r*sin(ang);
						int y_01 = (int) y - r*cos(ang);

						if ((x_01 >= 0 && x_01 < magnitude.rows) && (y_01 >= 0 && y_01 < magnitude.cols)) {
							parameterSpace[x_01][y_01][r] += 1;
						}
					}

				}
			}

		}
	}

	//create houghSpace
	Mat houghSpace = Mat(magnitude.rows, magnitude.cols, CV_32SC1, float(0));

	cvtColor(magnitude, magnitude, CV_GRAY2BGR);
	//draw best circle
	int maximum = 0;
	for (i = 0; i < dim1; ++i) {
		for (j = 0; j < dim2; ++j) {
			int houghSum = 0;
			for (k = minRad; k < maxRad; ++k) {

				houghSum += parameterSpace[i][j][k];
				if (parameterSpace[i][j][k] > maximum) {
					maximum = parameterSpace[i][j][k];
				}

			}
			houghSpace.at<int>(i, j) = houghSum;
		}
	}

	for (i = 0; i < dim1; ++i) {
		for (j = 0; j < dim2; ++j) {
			for (k = minRad; k < maxRad; ++k) {
				if ((maximum - parameterSpace[i][j][k]) < detectionThreshold) {
					tuple<int, int> currentCentre = make_tuple(j, i);
					if (find(bestCircles.begin(), bestCircles.end(), currentCentre) == bestCircles.end()) {
						bestCircles.push_back(currentCentre);
						circle(magnitude, Point(j, i), k, Scalar(255, 0, 0), 2);
					}
				}
			}
		}
	}

	double min, max;
	cv::minMaxLoc(houghSpace, &min, &max);
	Mat newHough;
	houghSpace.convertTo(newHough, CV_8U, 255.0 / (max - min), -255.0*min / (max - min));

	string houghName = "houghSpace" + imageName + ".jpg";
	imwrite(houghName, newHough);

	string magName = "magnitudeImage" + imageName + ".jpg";
	imwrite(magName, magnitude);

	return bestCircles;
}

vector<Rect> detectDartboards(Mat frame, Mat magnitude, Mat direction, string imageName, int detectionThreshold) {
	vector<Rect> dartboards;
	vector<Rect> improvedDartboards;
	Mat frame_gray;
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale(frame_gray, dartboards, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500, 500));

	// 3. Perform the Hough Transform to find the best fitting circles in the image.
	int maxDimension = (magnitude.rows > magnitude.cols) ? magnitude.rows : magnitude.cols;
	int maxRad = (maxDimension * 2) / 3, minRad = 20;
	int dim1 = magnitude.rows, dim2 = magnitude.cols;
	int houghThreshold = magnitude.cols / 24;
	vector<tuple<int, int> > bestCircles = houghSpace(magnitude, direction, 255, minRad, maxRad, imageName, houghThreshold);

	// 4. Print number of Dartboards found
	std::cout << "Detected dartboards: " << dartboards.size() << std::endl;

	// 5. Find the centre of each rectangles detected by Viola Jones.
	vector<tuple<int, int> > centrePoints;
	for (int i = 0; i < dartboards.size(); i++) {
		centrePoints.push_back(findCentreOfBox(dartboards[i]));
	}

	// 6. Draw box around dartboards found
	for (int j = 0; j < bestCircles.size(); j++) {
		tuple<int, int> bestCentre = bestCircles[j];
		for (int i = 0; i < dartboards.size(); i++) {
			tuple<int, int> currentCentre = centrePoints[i];
			if ((abs(get<0>(currentCentre) - get<0>(bestCentre)) < (detectionThreshold)) && (abs(get<1>(currentCentre) - get<1>(bestCentre)) < (detectionThreshold))) {
				Rect currentDartboard = dartboards[i];
				if (find(improvedDartboards.begin(), improvedDartboards.end(), currentDartboard) == improvedDartboards.end()) { //Ensures no two rectangles with the same coordinates are plotted twice.
					improvedDartboards.push_back(currentDartboard);
					rectangle(frame, Point(currentDartboard.x, currentDartboard.y), Point(currentDartboard.x + currentDartboard.width, currentDartboard.y + currentDartboard.height), Scalar(0, 255, 0), 2);
				}
			}
		}
	}

	string violaName = "improvedViola" + imageName + ".jpg";
	imwrite(violaName, frame);

	return improvedDartboards;
}

int mainViola(string imageName) {

	// LOADING THE IMAGE
	//string imageName;
	Mat image;
	Mat frame;

	//cin >> imageName;

	string toRead = "dart" + imageName + ".jpg";

	image = imread(toRead, CV_LOAD_IMAGE_COLOR);
	frame = imread(toRead, CV_LOAD_IMAGE_COLOR);

	if (!cascade.load(cascade_name)) { printf("--(!)Error loading\n"); return -1; };

	Mat newImageX(Size(image.cols, image.rows), CV_32FC1, Scalar(0));
	Mat newImageY(Size(image.cols, image.rows), CV_32FC1, Scalar(0));
	Mat magnitude(Size(image.cols, image.rows), CV_32FC1, Scalar(0));
	Mat direction(Size(image.cols, image.rows), CV_32FC1, Scalar(0));

	Mat x_direction = (Mat_<float>(3, 3) << 1, 0, -1,
											2, 0, -2,
											1, 0, -1);
	Mat y_direction = (Mat_<float>(3, 3) << 1, 2, 1,
											0, 0, 0,
											-1, -2, -1);

	Mat blurredMagnitude;
	GaussianBlur(image, blurredMagnitude, Size(5, 5), 0, 0, BORDER_DEFAULT);
	cvtColor(blurredMagnitude, image, CV_BGR2GRAY);

	int kernelRadiusX = 1;
	int kernelRadiusY = 1;

	for (int x = 0; x < image.rows; x++) {
		for (int y = 0; y < image.cols; y++) {
			float newValueX = 0.0;
			float newValueY = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {
					int imagex = (x + m + kernelRadiusX) % image.rows;
					int imagey = (y + n + kernelRadiusY) % image.cols;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					float imageval = (float)image.at<uchar>(imagex, imagey);
					float kernelX = (float)x_direction.at<float>(kernelx, kernely);
					float kernelY = (float)y_direction.at<float>(kernelx, kernely);

					// do the multiplication
					newValueX += (imageval * kernelX);
					newValueY += (imageval * kernelY);

				}
			}
			newImageX.at<float>(x, y) = (float)newValueX;
			newImageY.at<float>(x, y) = (float)newValueY;
			magnitude.at<float>(x, y) = (float)sqrt(pow(newValueX, 2) + pow(newValueY, 2));
			direction.at<float>(x, y) = (float)atan2((float)newValueY, (float)newValueX);
		}
	}

	normalize(newImageX, newImageX, 0, 255, CV_MINMAX);
	normalize(newImageY, newImageY, 0, 255, CV_MINMAX);
	normalize(magnitude, magnitude, 0, 255, CV_MINMAX);

	newImageX.convertTo(newImageX, CV_8UC1);
	newImageY.convertTo(newImageY, CV_8UC1);
	magnitude.convertTo(magnitude, CV_8UC1);

	for (int x = 0; x < magnitude.rows; x++) {
		for (int y = 0; y < magnitude.cols; y++) {
			uchar pixel = magnitude.at<uchar>(x, y);

			if (pixel > 20) {
				magnitude.at<uchar>(x, y) = 255;
			}
			else {
				magnitude.at<uchar>(x, y) = 0;
			}
		}
	}

	int detectionThreshold = magnitude.rows / 10;
	vector<Rect> detectedDartboards = detectDartboards(frame, magnitude, direction, imageName, detectionThreshold);

	vector<tuple<int, int>> groundTruth = findGroundTruth(stoi(imageName));

	float f1_score = f1Score(groundTruth, detectedDartboards, detectionThreshold);
	cout << f1_score << "\n";
	return 0;
}

int main() {

	/*for (int i = 0; i < 16; i++) {
		cout << "Dart " << i << "\n";
		stringstream ss;
		ss << i;
		string str = ss.str();
		mainViola(str);
	}*/

	int imageNumber;
	cin >> imageNumber;
	stringstream ss;
	ss << imageNumber;
	string str = ss.str();
	mainViola(str);
}