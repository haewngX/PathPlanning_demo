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

#define pi 3.14
using namespace std;
using namespace cv;

const double max_speed = 1.0; //[m/s] 最大速度
const double min_speed = -1.0; //[m/s] 最小速度
const double max_yawrate = 40.0 * pi / 180.0; //[rad/s] 最大角速度
const double max_accel = 0.2;  //[m/ss] 最大加速度
const double max_dyawrate = 40.0 * pi / 180.0; //[rad/ss] 最大角加速度
const double v_reso = 0.01;  //采样速度的步长
const double yawrate_reso = 0.1 * pi / 180.0;  //采样角速度的步长 [rad / s]
const double dt = 0.1;  
const double predict_time = 5.0;  //采样轨迹的时间
const double heading_cost_gain = 0.15; //目标函数中heading方位角评价函数的系数
const double speed_cost_gain = 1.0;//目标函数中速度评价函数的系数
const double obstacle_cost_gain = 1.0;  //目标函数中dist评价函数的系数
const double robot_radius = 1.0;  //机器人半径

Mat Map;  //地图为opencv生成的图片
int mapSize = 300;  //地图大小为300*300的正方形
using position = std::vector<std::array<float, 2>>; //存放xy坐标的vector，方便画图
using State = array<double, 5>; //状态
using Traj = vector<array<double, 5>>; //存放状态的轨迹
using Control = array<double, 2>; //速度
using Dynamic_Window = array<double, 4>; //速度的动态窗口

position start({ {0.0,0.0} });//起点坐标
position goal({ {100.0,100.0} });//终点坐标
// 添加障碍坐标
position ob({ {{-10.0, -10.0},
			   {0.0, 20.0},
			   {40.0, 20.0},
			   {50.0, 40.0},
			   {50.0, 50.0},
			   {50.0, 60.0},
			   {50.0, 90.0},
			   {80.0, 90.0},
			   {70.0, 90.0}} });

//初始化状态[x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
State x{ 0.0, 0.0, pi / 2.0, 0.0, 0.0 };

cv::Point2i cv_offset(float x, float y, int image_width, int image_height);
void drawMap(Traj ptraj); 
State motion(State x, Control u);
Dynamic_Window calc_dynamic_window(State x);
Traj predict_trajectory(State x, double v, double w);
Traj calc_final_input(State x, Control& u, Dynamic_Window dw);
double calc_to_goal_cost(Traj traj);
double calc_to_ob_cost(Traj traj);
Traj dwa_control(State x, Control& u);

int main()
{
	//初始化速度
	Control u{ 0.0, 0.0 };
	//初始化经历状态的轨迹
	Traj traj;
	traj.push_back(x);

	while (1)
	{
		Traj ptraj = dwa_control(x, u);
		x = motion(x, u);
		traj.push_back(x);
		//地图绘制
		drawMap(ptraj);
		imshow("画板", Map);
		waitKey(1);
		if (abs(x[0] - goal[0][0])<3 && abs(x[1] - goal[0][1]) < 3)
			break;
	}
	// 设置窗口
	Map = Mat::zeros(Size(500, 500), CV_8UC3);
	Map.setTo(255);              // 设置屏幕为白色
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
	//绘制轨迹
	for (int j = 0; j < traj.size(); j++) {
		cv::circle(Map, cv_offset(traj[j][0], traj[j][1], Map.cols, Map.rows),
			1, cv::Scalar(0, 0, 255), 1);
	}
	imshow("画板", Map);
	waitKey();
	return 0;
}

void drawMap(Traj ptraj) {
	// 设置窗口
	Map = Mat::zeros(Size(500, 500), CV_8UC3);
	Map.setTo(255);              // 设置屏幕为白色
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
	//绘制当前坐标 红色
	cv::circle(Map, cv_offset(x[0], x[1], Map.cols, Map.rows),
		2, cv::Scalar(0, 0, 255), 1);
	//绘制速度窗口轨迹
	for (int j = 0; j < ptraj.size(); j++) {
		cv::circle(Map, cv_offset(ptraj[j][0], ptraj[j][1], Map.cols, Map.rows),
			1, cv::Scalar(210, 0, 0), 1);
	}
}

//opencv图像的坐标和本程序中Map的坐标定义不同，用该函数把一个坐标点转换为图像上的点
cv::Point2i cv_offset(
	float x, float y, int image_width = 500, int image_height = 500) {
	cv::Point2i output;
	//output.x = int(x * 100) + image_width / 2;
	//output.y = image_height - int(y * 100) - image_height / 3;
	output.x = int(100 + x);
	output.y = int(image_height - y - 100);
	return output;
};

//根据速度计算下一时刻的状态
State motion(State x, Control u) {
	x[2] += u[1] * dt; //theta偏航角
	x[0] += u[0] * cos(x[2])*dt; //x坐标
	x[1] += u[0] * sin(x[2])*dt; //y坐标
	x[3] = u[0]; //线速度
	x[4] = u[1]; //角速度
	return x;
}

//计算速度的动态窗口
Dynamic_Window calc_dynamic_window(State x) {
	return { {max(min_speed,x[3] - max_accel * dt),
			  min(max_speed,x[3] + max_accel * dt),
			  max(-max_yawrate,x[4] - max_dyawrate * dt),
			  min(max_yawrate,x[4] + max_dyawrate * dt)} };
}

//根据速度预测一段时间内的轨迹
Traj predict_trajectory(State x, double v, double w) {
	Traj traj;
	traj.push_back(x);
	Control c = { v,w };
	double time = 0;
	for (double time = 0; time <= predict_time; time += dt)
	{
		x = motion(x, c);
		traj.push_back(x);
	}
	return traj;
}

//用动态窗口法计算出最优速度和轨迹
Traj calc_final_input(State x, Control& u, Dynamic_Window dw) {
	double min_cost = 1000000.0;
	Control best_u = { 0.0,0.0 };
	Traj best_traj;
	//评估动态窗口中每一条轨迹
	for (double v = dw[0]; v <= dw[1]; v += v_reso)
	{
		for (double w = dw[2]; w <= dw[3]; w += yawrate_reso)
		{
			Traj traj = predict_trajectory(x, v, w);
			//计算cost
			double to_goal_cost = heading_cost_gain * calc_to_goal_cost(traj);
			double speed_cost = speed_cost_gain * (max_speed - traj[traj.size() - 1][3]);
			double ob_cost = obstacle_cost_gain * calc_to_ob_cost(traj);
			double final_cost = to_goal_cost + speed_cost + ob_cost;
			if (min_cost >= final_cost) {
				min_cost = final_cost;
				best_u = { v,w };
				best_traj = traj;
			}
		}
	}
	u = best_u;
	return best_traj;
}

//计算到目标的cost
double calc_to_goal_cost(Traj traj){
	//轨迹终点的偏航角和机器人到目标点夹角之差
	double dx = goal[0][0] - traj[traj.size() - 1][0];
	double dy = goal[0][1] - traj[traj.size() - 1][1];
	double error_angle = atan2(dy, dx);
	double cost = abs(error_angle - traj[traj.size() - 1][2]);
	return cost;
}

//计算障碍物的cost
double calc_to_ob_cost(Traj traj) {
	//计算当前轨迹到障碍物的最小距离
	int skip_n = 2;
	double minr = 1000000.0;
	for (int i = 0; i < traj.size(); i += skip_n) 
	{
		for (int j = 0; j < ob.size(); j++) 
		{
			double dx = traj[i][0] - ob[j][0];
			double dy = traj[i][1] - ob[j][1];
			double r = sqrt(pow(dx, 2) + pow(dy, 2));
			if(r<= robot_radius)
				return 1000000.0;
			if (minr >= r)
				minr = r;
		}
	}
	return 1.0 / minr;
}

//动态窗口法求速度返回轨迹
Traj dwa_control(State x, Control& u) {
	Dynamic_Window dw = calc_dynamic_window(x);
	Traj traj = calc_final_input(x, u, dw);
	return traj;
}