#ifndef LINE_SURFACE_INTERSECTION_CUH
#define LINE_SURFACE_INTERSECTION_CUH

#include <cuda_runtime.h>
#include <vector>
#include <cmath>

using namespace std;

// GPU性能指标结构体
struct GPUPerformanceMetrics {
    float totalTime;           // 总耗时（ms）
    float memcpyH2D;          // Host到Device传输时间（ms）
    float kernelExecution;     // 内核执行时间（ms）
    float memcpyD2H;          // Device到Host传输时间（ms）
    int processedRays;         // 处理的射线数
    int validIntersections;    // 有效交点数
    int blockSize;             // 线程块大小
    int gridSize;              // 网格大小
    size_t deviceMemoryUsed;   // GPU显存使用量（字节）
};

// 三维点结构体
struct Point3D {
    float x, y, z;
    __host__ __device__ Point3D() : x(0), y(0), z(0) {}
    __host__ __device__ Point3D(float x, float y, float z) : x(x), y(y), z(z) {}
};

// 圆柱体位置结构体（用于铣削路径记录）
struct CylinderPosition {
    float x, y, z;  // 中心位置
    __host__ __device__ CylinderPosition() : x(0), y(0), z(0) {}
    __host__ __device__ CylinderPosition(float x, float y, float z) : x(x), y(y), z(z) {}
};


// 直线结构体
struct Line {
    Point3D origin;
    Point3D direction;
};

// 曲面函数声明
__device__ float surface_function(float x, float y, float z);

// 在GPU上计算单条直线与曲面的交点
__global__ void findIntersectionPoints(
    const Line* lines,
    Point3D* intersections,
    int* validFlags,
    int numLines,
    float stepSize,
    int maxSteps
);

// 主机端包装函数
void findIntersectionsGPU(
    const vector<Line>& lines,
    vector<Point3D>& intersections,
    vector<int>& validFlags,
    float stepSize = 0.01f,
    int maxSteps = 1000
);

// GPU并行布尔运算：多个圆柱体位置累积移除 - 长方体（带性能指标）
__global__ void computeBooleanSubtractionKernel(
    const Line* lines,
    Point3D* intersections,
    int* validFlags,
    Line* visibleLines,
    int numLines,
    int linesPerPlane,
    float cylinderRadius,
    float cylinderHeight,
    const CylinderPosition* cylinderPositions,
    int numCylinders,
    float boxWidth,
    float boxDepth,
    float boxHeight
);

// 主机端包装函数（带性能指标，支持多圆柱体路径）
void computeBooleanSubtractionGPU(
    const vector<Line>& lines,
    vector<Point3D>& intersections,
    vector<int>& validFlags,
    vector<Line>& visibleLines,
    int linesPerPlane,
    float cylinderRadius,
    float cylinderHeight,
    const vector<CylinderPosition>& cylinderPositions,
    float boxWidth,
    float boxDepth,
    float boxHeight,
    GPUPerformanceMetrics* metrics = nullptr
);

// 主机端包装函数：GPU布尔运算
void computeBooleanSubtractionGPU(
    const vector<Line>& lines,
    vector<Point3D>& intersections,
    vector<int>& validFlags,
    vector<Line>& visibleLines,
    int linesPerPlane,
    float cylinderRadius,
    float cylinderHeight,
    float boxWidth,
    float boxDepth,
    float boxHeight
);

// GPU数据持久化管理（优化性能，避免每帧重新分配和传输）
void initializeGPUData(const vector<Line>& lines);
void cleanupGPUData();
void computeBooleanSubtractionGPU_Optimized(
    vector<Point3D>& intersections,
    vector<int>& validFlags,
    vector<Line>& visibleLines,
    int linesPerPlane,
    float cylinderRadius,
    float cylinderHeight,
    const vector<CylinderPosition>& cylinderPositions,
    float boxWidth,
    float boxDepth,
    float boxHeight,
    GPUPerformanceMetrics* metrics = nullptr
);

#endif