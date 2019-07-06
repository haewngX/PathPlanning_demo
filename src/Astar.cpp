#include <iostream>
#include <Eigen/Eigen>
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

const int ALLOW_VERTEX_PASSTHROUGH = 1; //是否允许斜着走
const int NODE_FLAG_CLOSED = -1; //该节点在CLOSE中
const int NODE_FLAG_UNDEFINED = 0; //该节点没有遍历过
const int NODE_FLAG_OPEN = 1;  //该节点在OPEN中

const int NODE_TYPE_ZERO = 0;  //该节点可以通行
const int NODE_TYPE_OBSTACLE = 1;  //该节点有障碍
const int NODE_TYPE_START = 2;  //该节点为起点
const int NODE_TYPE_END = 3;  //该节点为终点

//每个像素对应地图中节点
class MapNode {
public:
	int x = -1; 
	int y = -1;
	int h = 0; //启发值
	int g = 0; //耗散值
	int type = NODE_TYPE_ZERO;
	int flag = NODE_FLAG_UNDEFINED;
	MapNode *parent = 0;

	MapNode() { }

	MapNode(int x, int y, int type = NODE_TYPE_ZERO, int flag = NODE_FLAG_UNDEFINED, MapNode *parent = 0) {
		this->x = x;
		this->y = y;
		this->type = type;
		this->flag = flag;
		this->parent = parent;
	}

	int f() {
		return g + h; //Astar
		return g; //只考虑耗散值，即为Dijkstra
	}
};

//地图为opencv生成的图片
Mat Map; 
//地图大小为300*300的正方形
int mapSize = 300; 
//地图中每个节点的信息 每个索引对应一个节点  具体转换方式为index = 300*y+x，mapData[index]即为坐标xy的节点Node
vector<MapNode> mapData; 
//Open表
vector<MapNode *> openList;  

MapNode *startNode; // 指向起点Node
MapNode *targetNode;// 指向终点Node

MapNode *mapAt(int x, int y); //将XY坐标转换为对应的索引

vector<MapNode *> neighbors(MapNode *node);//求当前节点的邻居节点

int computeH(MapNode *node1, MapNode *node2);//计算H

int computeG(MapNode *node1, MapNode *node2);//计算G

vector<MapNode *> Astar(); //寻找路径主函数

cv::Point2i cv_offset(float x, float y, int image_width, int image_height);

void drawMap(); //地图绘制

void drawPath(Mat &map, vector<MapNode *> path);  //绘制最后的路径

void drawOpenList(); //绘制Openlist中的点

using position = std::vector<std::array<float, 2>>; //存放xy坐标的vector，方便画图

int main()
{
	//地图绘制
	drawMap();
	//地图中每个节点的信息。300*300共有90000个节点
	mapData = vector<MapNode>(90000);
	//根据地图中每个点的像素获取该点的信息
	for (int y = 0; y < mapSize; y++) {
		for (int x = 0; x < mapSize; x++) {
			if (Map.at<Vec3b>(400 - y, 100 + x) == Vec3b(255, 255, 255)) {
				mapData[y * mapSize + x] = MapNode(x, y, NODE_TYPE_ZERO);
			}
			else if (Map.at<Vec3b>(400 - y, 100 + x) == Vec3b(0, 0, 0)) {
				mapData[y * mapSize + x] = MapNode(x, y, NODE_TYPE_OBSTACLE);
			}
			else if (Map.at<Vec3b>(400 - y, 100 + x) == Vec3b(0, 255, 0)) {
				MapNode node(x, y, NODE_TYPE_START);
				mapData[y * mapSize + x] = node;
				startNode = &mapData[y * 300 + x];
			}
			else if (Map.at<Vec3b>(400 - y, 100 + x) == Vec3b(255, 0, 0)) {
				MapNode node(x, y, NODE_TYPE_END);
				mapData[y * mapSize + x] = node;
				targetNode = &mapData[y * 300 + x];
			}
			else {
				Map.at<Vec3b>(400 - y, 100 + x) = Vec3b(0, 0, 0);
				mapData[y * mapSize + x] = MapNode(x, y, NODE_TYPE_OBSTACLE);
			}
		}
	}

	cout << "startNode=(" << startNode->x << ", " << startNode->y << ")" << endl;
	cout << "endNode=(" << targetNode->x << ", " << targetNode->y << ")" << endl;

	openList.push_back(startNode);//把起点放入openlist
	vector<MapNode *> path = Astar();//A*算法寻路
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
	position start({ {50.0,50.0} });//起点坐标
	position goal({ {250.0,250.0} });//终点坐标
	// 添加障碍坐标
	position ob({ {{0.0, 0.0}} });
	for (int i = 1; i < mapSize; i++)
	{
		ob.push_back({ 0.0, float(i) });
	}
	for (int i = 1; i < mapSize; i++)
	{
		ob.push_back({ 300.0, float(i) });
	}
	for (int i = 1; i < mapSize; i++)
	{
		ob.push_back({ float(i), 0.0 });
	}
	for (int i = 1; i < mapSize; i++)
	{
		ob.push_back({ float(i), 300.0 });
	}
	for (int i = 1; i < 200; i++)
	{
		ob.push_back({ 100, float(i) });
	}
	for (int i = 100; i < mapSize; i++)
	{
		ob.push_back({ 200, float(i) });
	}
	//绘制障碍 黑色
	for (unsigned int j = 0; j < ob.size(); j++) {
		cv::circle(Map, cv_offset(ob[j][0], ob[j][1], Map.cols, Map.rows),
			1, cv::Scalar(0, 0, 0), -2);
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
			2, cv::Scalar(0, 200, 155), 2);
		imshow("画板", Map);
		waitKey(1);
		cout << "->(" << node->x << "," << node->y << ")";
	}
	imshow("画板", Map);
	waitKey();
	cout << endl;
}

//绘制Openlist中的点
void drawOpenList() {
	for (int i = 0; i < openList.size(); i++) {
		MapNode *node = openList[i];
		if (node == startNode || node == targetNode)continue;
		cv::circle(Map, cv_offset(node->x, node->y, Map.cols, Map.rows),
			2, cv::Scalar(210, 210, 210), 2);
		imshow("画板", Map);
		waitKey(1);
	}
}

//A*算法
vector<MapNode *> Astar() {
	vector<MapNode *> path; //该数组存放路径上的节点
	cout << "Finding started!" << endl;
	int iteration = 0;
	MapNode *node;
	MapNode *reversedPtr = 0;
	while (openList.size() > 0) {
		node = openList.at(0);
		//找到openlist中f(n)最小的节点
		for (int i = 0, max = openList.size(); i < max; i++) {
			if ((openList[i]->f() + openList[i]->h)<= (node->f() + node->h)) {
				node = openList[i];
			}
		}
		//把这个节点从open表中移除，并且标记为Close
		openList.erase(remove(openList.begin(), openList.end(), node), openList.end());
		node->flag = NODE_FLAG_CLOSED;

		//如果到达了目标点，跳出循环，reversedPtr记为目标点
		if (node == targetNode) {
			cout << "Reached the target node." << endl;
			reversedPtr = node;
			break;
		}
		//找到当前节点的相邻节点
		vector<MapNode *> neighborNodes = neighbors(node);
		//遍历所有相邻节点
		for (int i = 0; i < neighborNodes.size(); i++) {
			MapNode *_node = neighborNodes[i];
			//如果相邻节点在close（已经访问过）中或者为障碍物，就跳过，判断下一个相邻节点
			if (_node->flag == NODE_FLAG_CLOSED || _node->type == NODE_TYPE_OBSTACLE) {
				continue;
			}
			int g = node->g + computeG(_node, node);//计算该相邻节点的耗散值
			//如果该相邻节点不在open中，就将其加入open
			if (_node->flag != NODE_FLAG_OPEN) {
				_node->g = g;
				_node->h = computeH(_node, targetNode);
				_node->parent = node;
				_node->flag = NODE_FLAG_OPEN;
				openList.push_back(_node);
			}
		}
		/*drawOpenList();*/
		if (openList.size() <= 0) break;
	}
	if (reversedPtr == 0) {
		cout << "Target node is unreachable." << endl;
	}
	//从终点逐步追踪parent节点到起点得出路径
	else {
		MapNode *_node = reversedPtr;
		while (_node->parent != 0) {
			path.push_back(_node);
			_node = _node->parent;
		}
		reverse(path.begin(), path.end());
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

	if (ALLOW_VERTEX_PASSTHROUGH) { //如果允许斜着走
		// 左下
		if ((_node = mapAt(node->x - 1, node->y - 1)) != 0)available.push_back(_node);
		// 右下
		if ((_node = mapAt(node->x + 1, node->y - 1)) != 0)available.push_back(_node);
		// 右上
		if ((_node = mapAt(node->x + 1, node->y + 1)) != 0)available.push_back(_node);
		// 左上
		if ((_node = mapAt(node->x - 1, node->y + 1)) != 0)available.push_back(_node);
	}

	return available;
}

//计算H值 节点到目标点的估计代价
int computeH(MapNode *node1, MapNode *node2) {
	//如果允许斜着走，计算欧氏距离
	if (ALLOW_VERTEX_PASSTHROUGH) {
		double x_distance = double(node2->x) - double(node1->x);
		double y_distance = double(node2->y) - double(node1->y);
		double distance2 = pow(x_distance, 2) + pow(y_distance, 2);
		return int(sqrt(distance2));
	}
	//否则计算曼哈顿距离
	else {
		return abs(node2->x - node1->x) + abs(node2->y - node1->y);
	}
}

//计算G值 从起点到当前点的实际代价
int computeG(MapNode *node1, MapNode *node2) {
	//如果允许斜着走，计算欧氏距离
	if (ALLOW_VERTEX_PASSTHROUGH) {
		double x_distance = double(node2->x) - double(node1->x);
		double y_distance = double(node2->y) - double(node1->y);
		double distance2 = pow(x_distance, 2) + pow(y_distance, 2);
		return int(sqrt(distance2));
	}
	//否则计算曼哈顿距离
	else {
		return abs(node2->x - node1->x) + abs(node2->y - node1->y);
	}
}

// 返回坐标所对应索引的节点
MapNode *mapAt(int x, int y) {
	if (x < 0 || y < 0 || x >= mapSize || y >= mapSize)return 0;
	return &mapData[y * mapSize + x];
}