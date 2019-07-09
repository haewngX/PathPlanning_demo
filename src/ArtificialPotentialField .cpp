#include <iostream>
#include <stdio.h>      
#include <math.h>    
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <array>
#include <vector>

using namespace std;
using namespace cv;

double KP = 5.0;  // 引力增益K
double ETA = 100000000.0; // 斥力增益η
const double robot_radius = 5.0;  //机器人半径
using position = std::vector<std::array<float, 2>>; //存放xy坐标的vector，方便画图
position start({ {0.0,0.0} });//起点坐标
position goal({ {250.0,250.0} });//终点坐标
// 添加障碍坐标
position ob({ {{150.0, 150.0},
			   {50.0, 50.0},
			   {100.0, 92.0},
			   {200.0, 198.0}} });

//每个像素对应地图中节点
class MapNode {
public:
	int x = -1; 
	int y = -1;
	double ug = 0.0; //引力
	double uo = 0.0; //斥力
	MapNode() { }

	MapNode(int x, int y,int ug, int uo) {
		this->x = x;
		this->y = y;
		this->ug = ug;
		this->uo = uo;
	}
	//计算合力
	double uf() {
		return ug + uo; 
	}
};

//地图为opencv生成的图片
Mat Map; 
//地图大小为300*300的正方形
int mapSize = 300; 
//地图中每个节点的信息 每个索引对应一个节点  具体转换方式为index = 300*y+x，mapData[index]即为坐标xy的节点Node
vector<MapNode> mapData; 

MapNode *startNode; // 指向起点Node
MapNode *targetNode;// 指向终点Node

MapNode *mapAt(int x, int y); //将XY坐标转换为对应的索引

vector<MapNode *> neighbors(MapNode *node);//求当前节点的邻居节点

cv::Point2i cv_offset(float x, float y, int image_width, int image_height);

void drawMap(); //地图绘制

void drawPath(Mat &map, vector<MapNode *> path);  //绘制最后的路径

double calc_attractive_potential(int x, int y); //计算引力
double calc_repulsive_potential(int x, int y); //计算斥力
vector<MapNode *> potential_field_planning();  //人工势场法

int main()
{
	//地图绘制
	drawMap();
	//地图中每个节点的信息。300*300共有90000个节点
	mapData = vector<MapNode>(90000);
	//获取地图中每个点的合力
	for (int y = 0; y < mapSize; y++) {
		for (int x = 0; x < mapSize; x++) {
			double ug = calc_attractive_potential(x, y);
			double uo = calc_repulsive_potential(x, y);
			mapData[y * mapSize + x] = MapNode(x, y, ug, uo);
		}	
	}
	startNode = &mapData[0 * 300 + 0];//起点
	targetNode = &mapData[250 * 300 + 250];//终点
	cout << "startNode=(" << startNode->x << ", " << startNode->y << ")" << endl;
	cout << "endNode=(" << targetNode->x << ", " << targetNode->y << ")" << endl;
	vector<MapNode *> path = potential_field_planning();//人工势场寻路
	drawPath(Map, path); //绘制最后的路径
	return 0;
}

////opencv图像的坐标和本程序中Map的坐标定义不同，用该函数把一个坐标点转换为图像上的点
cv::Point2i cv_offset(
	float x, float y, int image_width = 500, int image_height = 500) {
	cv::Point2i output;
	//output.x = int(x * 100) + image_width / 2;
	//output.y = image_height - int(y * 100) - image_height / 3;
	output.x = int(100 + x);
	output.y = int(image_height - y - 100);
	return output;
};

void drawMap() {
	// 设置窗口
	Map = Mat::zeros(Size(500, 500), CV_8UC3);
	Map.setTo(255);              // 设置屏幕为白色
	//绘制障碍 黑色
	for (unsigned int j = 0; j < ob.size(); j++) {
		cv::circle(Map, cv_offset(ob[j][0], ob[j][1], Map.cols, Map.rows),
			1, cv::Scalar(0, 0, 0), 5);
	}
	//绘制目标 蓝色
	cv::circle(Map, cv_offset(goal[0][0], goal[0][1], Map.cols, Map.rows),
		2, cv::Scalar(255, 0, 0), 2);
	//绘制起点 绿色
	cv::circle(Map, cv_offset(start[0][0], start[0][1], Map.cols, Map.rows),
		2, cv::Scalar(0, 255, 0), 2);
}

//绘制最终的路径
void drawPath(Mat &map, vector<MapNode *> path) {
	//Path存放了每一个路径上的点
	for (int i = 0; i < path.size() - 1; i++) {
		MapNode *node = path[i];
		cv::circle(Map, cv_offset(node->x, node->y, Map.cols, Map.rows),
			2, cv::Scalar(0, 200, 155), 1);
		imshow("画板", Map);
		waitKey(1);
		cout << "->(" << node->x << "," << node->y << ")";
	}
	imshow("画板", Map);
	waitKey();
	cout << endl;
}

//人工势场法
vector<MapNode *> potential_field_planning() {
	vector<MapNode *> path; //该数组存放路径上的节点
	cout << "Finding started!" << endl;
	MapNode *node = startNode;
	path.push_back(node);
	MapNode *next;
	while (1)
	{
		double minp = 1000000.0;
		//如果到达了目标点，跳出循环，
		if (abs(node->x-targetNode->x)<2&& abs(node->y - targetNode->y) < 2)
		{
			cout << "Reached the target node." << endl;
			break;
		}
		//找到当前节点的相邻节点
		vector<MapNode *> neighborNodes = neighbors(node);
		//遍历所有相邻节点，选择合力方向
		for (int i = 0; i < neighborNodes.size(); i++) 
		{
			MapNode *_node = neighborNodes[i];
			double p = _node->uf();
			if (minp > p)
			{
				minp = p;
				next = _node;
			}
		}
		path.push_back(next);
		node = next;
		Map = Mat::zeros(Size(500, 500), CV_8UC3);
		Map.setTo(255);              // 设置屏幕为白色
		//绘制障碍 黑色
		for (unsigned int j = 0; j < ob.size(); j++) {
			cv::circle(Map, cv_offset(ob[j][0], ob[j][1], Map.cols, Map.rows),
				1, cv::Scalar(0, 0, 0), 2);
		}
		//绘制目标 蓝色
		cv::circle(Map, cv_offset(goal[0][0], goal[0][1], Map.cols, Map.rows),
			2, cv::Scalar(255, 0, 0), 2);
		//绘制起点 绿色
		cv::circle(Map, cv_offset(start[0][0], start[0][1], Map.cols, Map.rows),
			2, cv::Scalar(0, 255, 0), 2);
		//绘制机器人 红
		cv::circle(Map, cv_offset(node->x, node->y, Map.cols, Map.rows),
			2, cv::Scalar(0, 0, 255), 2);
		imshow("画板", Map);
		waitKey(20);
	}
	//返回路径
	return path;
}

// 返回邻居节点
vector<MapNode *> neighbors(MapNode *node) {
	vector<MapNode *> available;
	MapNode *_node;
	//如果该节点的邻居在地图中，那么返回
	// 左
	if ((_node = mapAt(node->x - 1, node->y)) != 0)available.push_back(_node);
	// 下
	if ((_node = mapAt(node->x, node->y - 1)) != 0)available.push_back(_node);
	// 右
	if ((_node = mapAt(node->x + 1, node->y)) != 0)available.push_back(_node);
	// 上
	if ((_node = mapAt(node->x, node->y + 1)) != 0)available.push_back(_node);
	// 左下
	if ((_node = mapAt(node->x - 1, node->y - 1)) != 0)available.push_back(_node);
	// 右下
	if ((_node = mapAt(node->x + 1, node->y - 1)) != 0)available.push_back(_node);
	// 右上
	if ((_node = mapAt(node->x + 1, node->y + 1)) != 0)available.push_back(_node);
	// 左上
	if ((_node = mapAt(node->x - 1, node->y + 1)) != 0)available.push_back(_node);
	return available;
}

// 返回坐标所对应索引的节点
MapNode *mapAt(int x, int y) {
	if (x < 0 || y < 0 || x >= mapSize || y >= mapSize)return 0;
	return &mapData[y * mapSize + x];
}

//计算引力
double calc_attractive_potential(int x, int y) {
	return 0.5*KP*(sqrt(pow((x - goal[0][0]), 2) + pow((y - goal[0][1]), 2)));
}

//计算斥力
double calc_repulsive_potential(int x, int y) {
	double dmin = 1000000.0;//距离机器人最近的障碍物距离
	for (int i = 0; i < ob.size(); i++)
	{
		double d = sqrt(pow((x - ob[i][0]), 2) + pow((y - ob[i][1]), 2));
		if (dmin >= d)
			dmin = d;
	}
	if (dmin <= robot_radius)
	{
		if (dmin <= 0.1)
			dmin = 0.1;
		return 0.5*ETA*pow((1.0 / dmin - 1.0 / robot_radius), 2);
	}
	else
		return 0.0;
}