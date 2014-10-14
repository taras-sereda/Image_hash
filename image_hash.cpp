#include <stdio.h>
#include <dirent.h>
#include <vector>
#include <string>
#include <bitset>
#include <cv.h>
#include <highgui.h>
#include <inttypes.h>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

Mat posterize(Mat src, int clusters) {

	Mat samples(src.rows * src.cols, 3, CV_32F);
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			for (int z = 0; z < 3; z++)
				samples.at<float> (y + x * src.rows, z)
						= src.at<Vec3b> (y, x)[z];

	int clusterCount = clusters;
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER
			| CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS,
			centers);

	Mat new_image(src.size(), src.type());
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++) {
			int cluster_idx = labels.at<int> (y + x * src.rows, 0);
			new_image.at<Vec3b> (y, x)[0] = centers.at<float> (cluster_idx, 0);
			new_image.at<Vec3b> (y, x)[1] = centers.at<float> (cluster_idx, 1);
			new_image.at<Vec3b> (y, x)[2] = centers.at<float> (cluster_idx, 2);
		}
	return new_image;
}

uint64_t avg_hash(Mat im) {
	Mat im_gray;
	Mat resized_img;
	Mat blured_img;
	Mat im_post;

	medianBlur(im, im_post, 7);

	resize(im_post, resized_img, Size(8, 8), 0, 0, INTER_AREA);
	cvtColor(resized_img, im_gray, CV_BGR2GRAY);

	int averageValue = 0;
	Size sz = im_gray.size();

	int h = sz.height;
	int w = sz.width;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			averageValue += (int) im_gray.at<uchar> (j, i);
		}
	}
	averageValue /= (int) h * w;
	uint64_t result = 0;

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			result <<= 1;
			result |= 1 * ((int) im_gray.at<uchar> (j, i) >= averageValue);
		}
	}
	return result;
}

// autocrop function
bool is_border(cv::Mat& edge, cv::Vec3b color) {
	cv::Mat im = edge.clone().reshape(0, 1);

	bool res = true;
	for (int i = 0; i < im.cols; ++i)
		res &= (color == im.at<cv::Vec3b> (0, i));

	return res;
}

/**
 * Function to auto-cropping image
 *
 * Parameters:
 *   src   The source image
 *   dst   The destination image
 */
void autocrop(cv::Mat& src, cv::Mat& dst) {
	cv::Rect win(0, 0, src.cols, src.rows);
	cv::Mat post_src = posterize(src, 4);
	std::vector<cv::Rect> edges;
	edges.push_back(cv::Rect(0, 0, post_src.cols, 1));
	edges.push_back(cv::Rect(post_src.cols - 2, 0, 1, post_src.rows));
	edges.push_back(cv::Rect(0, post_src.rows - 2, post_src.cols, 1));
	edges.push_back(cv::Rect(0, 0, 1, post_src.rows));

	cv::Mat edge;
	int nborder = 0;
	cv::Vec3b color = post_src.at<cv::Vec3b> (0, 0);

	for (int i = 0; i < (int) edges.size(); ++i) {
		edge = post_src(edges[i]);
		nborder += is_border(edge, color);
	}

	if (nborder < 4) {
		post_src.copyTo(dst);
		return;
	}

	bool next;

	do {
		edge = post_src(cv::Rect(win.x, win.height - 2, win.width, 1));
		if ((next = is_border(edge, color)))
			win.height--;
	} while (next && win.height > 0);

	do {
		edge = post_src(cv::Rect(win.width - 2, win.y, 1, win.height));
		if ((next = is_border(edge, color)))
			win.width--;
	} while (next && win.width > 0);

	do {
		edge = post_src(cv::Rect(win.x, win.y, win.width, 1));
		if ((next = is_border(edge, color)))
			win.y++, win.height--;
	} while (next && win.y <= src.rows);

	do {
		edge = post_src(cv::Rect(win.x, win.y, 1, win.height));
		if ((next = is_border(edge, color)))
			win.x++, win.width--;
	} while (next && win.x <= post_src.cols);

	dst = src(win);
}

int main(int argc, char **argv) {



	if (argc < 2) {
		printf("no input args\n");
		printf("expected arg [dir name]\n");
		exit(1);
	}

	const char *dir_name = argv[1];
	DIR *dir = opendir(dir_name);
	struct dirent *dir_entry;

	char path[100];
	path[0] = '\0';
	Mat dst;
	uint64_t ahash;

	while ((dir_entry = readdir(dir)) != 0) {
		if (strcmp(dir_entry->d_name, ".") && strcmp(dir_entry->d_name, "..")) {
			strcat(path, dir_name);
			strcat(path, "/");
			strcat(path, dir_entry->d_name);
			Mat src = imread(path);
			if (!src.data)
				return -1;
			autocrop(src, dst);
			ahash = avg_hash(dst);
			cout << path << " hash: " << ahash << endl;
		}
		path[0] = '\0';
	}

	return 0;
}
