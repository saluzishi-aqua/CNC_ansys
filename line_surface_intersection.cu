#include "line_surface_intersection.cuh"
#include <iostream>
#include <chrono>

using namespace std;

// 曲面函数实现

__device__ float cylinder_sdf(float x, float y, float z) {
    // 圆柱体方程: x² + y² = radius², 且 -height/2 <= z <= height/2
    const float radius = 3.0f;      // 与主程序保持一致
    const float height = 10.0f;     // 与主程序保持一致

    // 计算点到圆柱体侧面的距离
    float side_distance = sqrtf(x * x + y * y) - radius;

    // 计算点到圆柱体顶面/底面的距离
    float top_distance = fabsf(z) - height / 2.0f;

    // 使用有向距离场，返回0表示在表面上
    // 正值表示在外部，负值表示在内部
    float cylinder_distance = fmaxf(side_distance, top_distance);

    return cylinder_distance;
}

__device__ float box_sdf(float x, float y, float z) {
    // 长方体距离场函数
    const float width = 8.0f;   // X方向
    const float depth = 8.0f;   // Y方向
    const float height = 8.0f;  // Z方向
    
    float dx = fabsf(x) - width / 2.0f;
    float dy = fabsf(y) - depth / 2.0f;
    float dz = fabsf(z) - height / 2.0f;
    
    // 外部距离
    float outside = sqrtf(
        fmaxf(dx, 0.0f) * fmaxf(dx, 0.0f) +
        fmaxf(dy, 0.0f) * fmaxf(dy, 0.0f) +
        fmaxf(dz, 0.0f) * fmaxf(dz, 0.0f)
    );
    
    // 内部距离
    float inside = fminf(fmaxf(dx, fmaxf(dy, dz)), 0.0f);
    
    return outside + inside;
}

__device__ float surface_function(float x, float y, float z) {
    // 返回圆柱体的SDF（用于向后兼容）
    return cylinder_sdf(x, y, z);
}


__global__ void findIntersectionPoints(
    const Line* lines,
    Point3D* intersections,
    int* validFlags,
    int numLines,
    float stepSize,
    int maxSteps) {

    // 使用CUDA内置变量 - 这些只能在设备代码中使用
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numLines) return;

    Line line = lines[idx];
    Point3D current = line.origin;

    // 标准化方向向量
    float length = sqrtf(
        line.direction.x * line.direction.x +
        line.direction.y * line.direction.y +
        line.direction.z * line.direction.z
    );

    // 避免除零错误
    if (length < 1e-6f) {
        intersections[idx] = Point3D(0, 0, 0);
        validFlags[idx] = 0;
        return;
    }

    Point3D dir = {
        line.direction.x / length,
        line.direction.y / length,
        line.direction.z / length
    };

    // 沿着直线搜索交点
    float prev_value = surface_function(current.x, current.y, current.z);
    bool found = false;

    for (int step = 0; step < maxSteps; ++step) {
        current.x += dir.x * stepSize;
        current.y += dir.y * stepSize;
        current.z += dir.z * stepSize;

        float current_value = surface_function(current.x, current.y, current.z);

        // 检查符号变化（表示穿过曲面）
        if (prev_value * current_value <= 0.0f && fabsf(prev_value - current_value) > 1e-6f) {
            // 使用二分法精确化交点
            Point3D low = {
                current.x - dir.x * stepSize,
                current.y - dir.y * stepSize,
                current.z - dir.z * stepSize
            };
            Point3D high = current;
            float low_value = prev_value;
            float high_value = current_value;

            for (int refine = 0; refine < 10; ++refine) {
                Point3D mid = {
                    (low.x + high.x) * 0.5f,
                    (low.y + high.y) * 0.5f,
                    (low.z + high.z) * 0.5f
                };

                float mid_value = surface_function(mid.x, mid.y, mid.z);

                if (low_value * mid_value <= 0.0f) {
                    high = mid;
                    high_value = mid_value;
                }
                else {
                    low = mid;
                    low_value = mid_value;
                }
            }

            intersections[idx] = Point3D(
                (low.x + high.x) * 0.5f,
                (low.y + high.y) * 0.5f,
                (low.z + high.z) * 0.5f
            );
            validFlags[idx] = 1;
            found = true;
            break;
        }

        prev_value = current_value;
    }

    if (!found) {
        intersections[idx] = Point3D(0, 0, 0);
        validFlags[idx] = 0;
    }
}

void findIntersectionsGPU(
    const vector<Line>& lines,
    vector<Point3D>& intersections,
    vector<int>& validFlags,
    float stepSize,
    int maxSteps) {

    int numLines = lines.size();
    intersections.resize(numLines);
    validFlags.resize(numLines);

    // 分配设备内存
    Line* d_lines = nullptr;
    Point3D* d_intersections = nullptr;
    int* d_validFlags = nullptr;

    cudaError_t err;

    // 分配GPU内存
    err = cudaMalloc(&d_lines, numLines * sizeof(Line));
    if (err != cudaSuccess) {
        cerr << "cudaMalloc failed for d_lines: " << cudaGetErrorString(err) << endl;
        return;
    }

    err = cudaMalloc(&d_intersections, numLines * sizeof(Point3D));
    if (err != cudaSuccess) {
        cerr << "cudaMalloc failed for d_intersections: " << cudaGetErrorString(err) << endl;
        cudaFree(d_lines);
        return;
    }

    err = cudaMalloc(&d_validFlags, numLines * sizeof(int));
    if (err != cudaSuccess) {
        cerr << "cudaMalloc failed for d_validFlags: " << cudaGetErrorString(err) << endl;
        cudaFree(d_lines);
        cudaFree(d_intersections);
        return;
    }

    // 复制数据到设备
    err = cudaMemcpy(d_lines, lines.data(), numLines * sizeof(Line), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy failed for d_lines: " << cudaGetErrorString(err) << endl;
        cudaFree(d_lines);
        cudaFree(d_intersections);
        cudaFree(d_validFlags);
        return;
    }

    // 计算网格和块大小
    int blockSize = 256;
    int gridSize = (numLines + blockSize - 1) / blockSize;

    cout << "启动CUDA内核: gridSize=" << gridSize << ", blockSize=" << blockSize << endl;

    // 启动内核
    auto start = chrono::high_resolution_clock::now();

    findIntersectionPoints << <gridSize, blockSize >> > (
        d_lines, d_intersections, d_validFlags, numLines, stepSize, maxSteps
        );

    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    // 检查CUDA错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
    }

    // 复制结果回主机
    err = cudaMemcpy(intersections.data(), d_intersections, numLines * sizeof(Point3D), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy failed for intersections: " << cudaGetErrorString(err) << endl;
    }

    err = cudaMemcpy(validFlags.data(), d_validFlags, numLines * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "cudaMemcpy failed for validFlags: " << cudaGetErrorString(err) << endl;
    }

    // 释放设备内存
    cudaFree(d_lines);
    cudaFree(d_intersections);
    cudaFree(d_validFlags);

    cout << "GPU计算完成，耗时: " << duration.count() << " 微秒" << endl;
    cout << "约 " << duration.count() / 1000.0 << " 毫秒" << endl;
}

// ========== GPU并行布尔运算实现 ==========

// Device函数：判断点是否在长方体内
__device__ bool isInsideBox(float x, float y, float z, float boxWidth, float boxDepth, float boxHeight) {
    return (fabsf(x) <= boxWidth / 2.0f) && 
           (fabsf(y) <= boxDepth / 2.0f) && 
           (fabsf(z) <= boxHeight / 2.0f);
}

// Device函数：计算射线与长方体的交点参数t
__device__ void computeBoxIntersection(
    const Line& line,
    float boxWidth, float boxDepth, float boxHeight,
    float* t_values, int* t_count
) {
    *t_count = 0;
    float temp_t[6];
    int temp_count = 0;
    
    // X面
    if (fabsf(line.direction.x) > 1e-6f) {
        temp_t[temp_count++] = (-boxWidth/2.0f - line.origin.x) / line.direction.x;
        temp_t[temp_count++] = (boxWidth/2.0f - line.origin.x) / line.direction.x;
    }
    // Y面
    if (fabsf(line.direction.y) > 1e-6f) {
        temp_t[temp_count++] = (-boxDepth/2.0f - line.origin.y) / line.direction.y;
        temp_t[temp_count++] = (boxDepth/2.0f - line.origin.y) / line.direction.y;
    }
    // Z面
    if (fabsf(line.direction.z) > 1e-6f) {
        temp_t[temp_count++] = (-boxHeight/2.0f - line.origin.z) / line.direction.z;
        temp_t[temp_count++] = (boxHeight/2.0f - line.origin.z) / line.direction.z;
    }
    
    // 验证哪些t值对应真正的相交
    for (int i = 0; i < temp_count; ++i) {
        float t = temp_t[i];
        float px = line.origin.x + t * line.direction.x;
        float py = line.origin.y + t * line.direction.y;
        float pz = line.origin.z + t * line.direction.z;
        
        if (isInsideBox(px, py, pz, boxWidth, boxDepth, boxHeight)) {
            t_values[(*t_count)++] = t;
            if (*t_count >= 2) break;  // 最多2个交点
        }
    }
    
    // 排序t值
    if (*t_count == 2 && t_values[0] > t_values[1]) {
        float swap = t_values[0];
        t_values[0] = t_values[1];
        t_values[1] = swap;
    }
}

// GPU Kernel：并行计算布尔减法
__global__ void computeBooleanSubtractionKernel(
    const Line* lines,
    Point3D* intersections,
    int* validFlags,
    Line* visibleLines,
    int numLines,
    int linesPerPlane,
    float cylinderRadius,
    float cylinderHeight,
    float boxWidth,
    float boxDepth,
    float boxHeight
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numLines) return;
    
    const Line& L = lines[idx];
    
    // 初始化为无效
    validFlags[idx] = 0;
    intersections[idx] = Point3D(0, 0, 0);
    visibleLines[idx] = Line{Point3D(0,0,0), Point3D(0,0,0)};
    
    // 计算与长方体的交点
    float box_t[2];
    int box_count = 0;
    computeBoxIntersection(L, boxWidth, boxDepth, boxHeight, box_t, &box_count);
    
    // 根据射线方向计算与圆柱体的交线段
    float cyl_t_start = 0.0f, cyl_t_end = 0.0f;
    bool has_cylinder = false;
    
    if (idx < linesPerPlane) {
        // Z方向线：x,y常量，z变化
        float x = L.origin.x;
        float y = L.origin.y;
        float r_sq = x*x + y*y;
        
        if (r_sq <= cylinderRadius * cylinderRadius) {
            cyl_t_start = -cylinderHeight / 2.0f;
            cyl_t_end = cylinderHeight / 2.0f;
            has_cylinder = true;
        }
    } else if (idx < 2 * linesPerPlane) {
        // Y方向线：x,z常量，y变化
        float x = L.origin.x;
        float z = L.origin.z;
        
        if (x*x <= cylinderRadius * cylinderRadius && fabsf(z) <= cylinderHeight / 2.0f) {
            float delta = sqrtf(cylinderRadius * cylinderRadius - x*x);
            cyl_t_start = -delta;
            cyl_t_end = delta;
            has_cylinder = true;
        }
    } else {
        // X方向线：y,z常量，x变化
        float y = L.origin.y;
        float z = L.origin.z;
        
        if (y*y <= cylinderRadius * cylinderRadius && fabsf(z) <= cylinderHeight / 2.0f) {
            float delta = sqrtf(cylinderRadius * cylinderRadius - y*y);
            cyl_t_start = -delta;
            cyl_t_end = delta;
            has_cylinder = true;
        }
    }
    
    if (!has_cylinder) return;
    
    // 布尔减法：圆柱体 - 长方体
    float result_start = cyl_t_start;
    float result_end = cyl_t_end;
    
    if (box_count == 2) {
        float box_start = box_t[0];
        float box_end = box_t[1];
        
        // 长方体完全覆盖圆柱体段
        if (box_start <= cyl_t_start && box_end >= cyl_t_end) {
            return;  // 完全被减去
        }
        // 长方体与圆柱体有重叠
        else if (box_start < cyl_t_end && box_end > cyl_t_start) {
            // 取第一段（长方体前面的部分）
            if (box_start > cyl_t_start) {
                result_end = box_start;
            } else if (box_end < cyl_t_end) {
                result_start = box_end;
            } else {
                return;  // 被完全覆盖
            }
        }
    }
    
    if (result_end > result_start) {
        validFlags[idx] = 1;
        float t_mid = (result_start + result_end) / 2.0f;
        
        if (idx < linesPerPlane) {
            // Z方向
            visibleLines[idx].origin = Point3D(L.origin.x, L.origin.y, result_start);
            visibleLines[idx].direction = Point3D(0, 0, result_end - result_start);
            intersections[idx] = Point3D(L.origin.x, L.origin.y, t_mid);
        } else if (idx < 2 * linesPerPlane) {
            // Y方向
            visibleLines[idx].origin = Point3D(L.origin.x, result_start, L.origin.z);
            visibleLines[idx].direction = Point3D(0, result_end - result_start, 0);
            intersections[idx] = Point3D(L.origin.x, t_mid, L.origin.z);
        } else {
            // X方向
            visibleLines[idx].origin = Point3D(result_start, L.origin.y, L.origin.z);
            visibleLines[idx].direction = Point3D(result_end - result_start, 0, 0);
            intersections[idx] = Point3D(t_mid, L.origin.y, L.origin.z);
        }
    }
}

// 主机端包装函数
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
) {
    int numLines = lines.size();
    
    // 分配设备内存
    Line* d_lines;
    Point3D* d_intersections;
    int* d_validFlags;
    Line* d_visibleLines;
    
    cudaMalloc(&d_lines, numLines * sizeof(Line));
    cudaMalloc(&d_intersections, numLines * sizeof(Point3D));
    cudaMalloc(&d_validFlags, numLines * sizeof(int));
    cudaMalloc(&d_visibleLines, numLines * sizeof(Line));
    
    // 复制输入数据到设备
    cudaMemcpy(d_lines, lines.data(), numLines * sizeof(Line), cudaMemcpyHostToDevice);
    
    // 配置kernel启动参数
    int blockSize = 256;
    int gridSize = (numLines + blockSize - 1) / blockSize;
    
    // 启动kernel
    computeBooleanSubtractionKernel<<<gridSize, blockSize>>>(
        d_lines, d_intersections, d_validFlags, d_visibleLines,
        numLines, linesPerPlane,
        cylinderRadius, cylinderHeight,
        boxWidth, boxDepth, boxHeight
    );
    
    cudaDeviceSynchronize();
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
    }
    
    // 复制结果回主机
    cudaMemcpy(intersections.data(), d_intersections, numLines * sizeof(Point3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(validFlags.data(), d_validFlags, numLines * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(visibleLines.data(), d_visibleLines, numLines * sizeof(Line), cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_lines);
    cudaFree(d_intersections);
    cudaFree(d_validFlags);
    cudaFree(d_visibleLines);
}