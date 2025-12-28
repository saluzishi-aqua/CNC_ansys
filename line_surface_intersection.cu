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
    
    // 对长方体尺寸轻微内缩，避免边界噪点
    const float BOX_SHRINK = 0.02f;
    float widthHalf = boxWidth/2.0f - BOX_SHRINK;
    float depthHalf = boxDepth/2.0f - BOX_SHRINK;
    float heightHalf = boxHeight/2.0f - BOX_SHRINK;
    
    // X面
    if (fabsf(line.direction.x) > 1e-6f) {
        temp_t[temp_count++] = (-widthHalf - line.origin.x) / line.direction.x;
        temp_t[temp_count++] = (widthHalf - line.origin.x) / line.direction.x;
    }
    // Y面
    if (fabsf(line.direction.y) > 1e-6f) {
        temp_t[temp_count++] = (-depthHalf - line.origin.y) / line.direction.y;
        temp_t[temp_count++] = (depthHalf - line.origin.y) / line.direction.y;
    }
    // Z面
    if (fabsf(line.direction.z) > 1e-6f) {
        temp_t[temp_count++] = (-heightHalf - line.origin.z) / line.direction.z;
        temp_t[temp_count++] = (heightHalf - line.origin.z) / line.direction.z;
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

// GPU Kernel：并行计算布尔减法（支持多个圆柱体累积移除）
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
    
    if (box_count != 2) return;  // 射线未穿过长方体
    
    // 收集所有圆柱体位置与射线的交集
    const int MAX_SEGMENTS = 2000;  // 最大线段数（增加到2000以支持更多路径点）
    float removed_starts[MAX_SEGMENTS];
    float removed_ends[MAX_SEGMENTS];
    int removed_count = 0;
    
    // 遍历所有圆柱体位置
    for (int c = 0; c < numCylinders && removed_count < MAX_SEGMENTS; ++c) {
        float cylinderCenterX = cylinderPositions[c].x;
        float cylinderCenterY = cylinderPositions[c].y;
        float cylinderCenterZ = cylinderPositions[c].z;
        
        float cyl_t_start = 0.0f, cyl_t_end = 0.0f;
        bool has_intersection = false;
        
        if (idx < linesPerPlane) {
            // Z方向线：x,y常量，z变化，direction=(0,0,1)
            float x = L.origin.x - cylinderCenterX;
            float y = L.origin.y - cylinderCenterY;
            float r_sq = x*x + y*y;
            
            const float RADIUS_SHRINK = 0.01f;  // 轻微收缩半径避免边缘噪点
            float radiusCheck = cylinderRadius - RADIUS_SHRINK;
            if (r_sq < radiusCheck * radiusCheck) {  // 使用<而不是<=
                // 圆柱体Z范围的实际坐标
                float z_min = cylinderCenterZ - cylinderHeight / 2.0f;
                float z_max = cylinderCenterZ + cylinderHeight / 2.0f;
                // 转换为参数t（t = (z - origin.z) / direction.z，但direction.z = 1）
                cyl_t_start = z_min - L.origin.z;
                cyl_t_end = z_max - L.origin.z;
                has_intersection = true;
            }
        } else if (idx < 2 * linesPerPlane) {
            // Y方向线：x,z常量，y变化，direction=(0,1,0)
            float x = L.origin.x - cylinderCenterX;
            float z = L.origin.z - cylinderCenterZ;
            
            if (fabsf(z) < cylinderHeight / 2.0f) {  // 使用<
                const float RADIUS_SHRINK = 0.01f;
                float radiusCheck = cylinderRadius - RADIUS_SHRINK;
                float r_sq = radiusCheck * radiusCheck - x*x;
                
                if (r_sq >= 0) {
                    float delta = sqrtf(r_sq);
                    // 圆柱体Y范围的实际坐标
                    float y_min = cylinderCenterY - delta;
                    float y_max = cylinderCenterY + delta;
                    // 转换为参数t（t = (y - origin.y) / direction.y，但direction.y = 1）
                    cyl_t_start = y_min - L.origin.y;
                    cyl_t_end = y_max - L.origin.y;
                    has_intersection = true;
                }
            }
        } else {
            // X方向线：y,z常量，x变化，direction=(1,0,0)
            float y = L.origin.y - cylinderCenterY;
            float z = L.origin.z - cylinderCenterZ;
            
            if (fabsf(z) < cylinderHeight / 2.0f) {  // 使用<
                const float RADIUS_SHRINK = 0.01f;
                float radiusCheck = cylinderRadius - RADIUS_SHRINK;
                float r_sq = radiusCheck * radiusCheck - y*y;
                
                if (r_sq >= 0) {
                    float delta = sqrtf(r_sq);
                    // 圆柱体X范围的实际坐标
                    float x_min = cylinderCenterX - delta;
                    float x_max = cylinderCenterX + delta;
                    // 转换为参数t（t = (x - origin.x) / direction.x，但direction.x = 1）
                    cyl_t_start = x_min - L.origin.x;
                    cyl_t_end = x_max - L.origin.x;
                    has_intersection = true;
                }
            }
        }
        
        // 如果有交集，添加到移除列表
        if (has_intersection) {
            removed_starts[removed_count] = cyl_t_start;
            removed_ends[removed_count] = cyl_t_end;
            removed_count++;
        }
    }
    
    // 如果没有圆柱体与射线相交，显示完整的长方体线段
    if (removed_count == 0) {
        validFlags[idx] = 1;
        float box_start = box_t[0];
        float box_end = box_t[1];
        
        if (idx < linesPerPlane) {
            // Z方向：t就是z坐标（因为direction是(0,0,1)，origin.z是起点）
            float z_start = L.origin.z + box_start * L.direction.z;
            float z_end = L.origin.z + box_end * L.direction.z;
            visibleLines[idx].origin = Point3D(L.origin.x, L.origin.y, z_start);
            visibleLines[idx].direction = Point3D(0, 0, z_end - z_start);
            intersections[idx] = Point3D(L.origin.x, L.origin.y, (z_start + z_end) / 2.0f);
        } else if (idx < 2 * linesPerPlane) {
            // Y方向：计算实际y坐标
            float y_start = L.origin.y + box_start * L.direction.y;
            float y_end = L.origin.y + box_end * L.direction.y;
            visibleLines[idx].origin = Point3D(L.origin.x, y_start, L.origin.z);
            visibleLines[idx].direction = Point3D(0, y_end - y_start, 0);
            intersections[idx] = Point3D(L.origin.x, (y_start + y_end) / 2.0f, L.origin.z);
        } else {
            // X方向：计算实际x坐标
            float x_start = L.origin.x + box_start * L.direction.x;
            float x_end = L.origin.x + box_end * L.direction.x;
            visibleLines[idx].origin = Point3D(x_start, L.origin.y, L.origin.z);
            visibleLines[idx].direction = Point3D(x_end - x_start, 0, 0);
            intersections[idx] = Point3D((x_start + x_end) / 2.0f, L.origin.y, L.origin.z);
        }
        return;
    }
    
    // 合并所有移除区间
    // 简单排序（冒泡排序，因为数量不多）
    for (int i = 0; i < removed_count - 1; ++i) {
        for (int j = 0; j < removed_count - 1 - i; ++j) {
            if (removed_starts[j] > removed_starts[j + 1]) {
                float temp_s = removed_starts[j];
                float temp_e = removed_ends[j];
                removed_starts[j] = removed_starts[j + 1];
                removed_ends[j] = removed_ends[j + 1];
                removed_starts[j + 1] = temp_s;
                removed_ends[j + 1] = temp_e;
            }
        }
    }
    
    // 合并重叠区间（添加epsilon容差处理浮点精度）
    const float EPSILON = 0.0001f;  // 浮点精度容差
    float merged_starts[MAX_SEGMENTS];
    float merged_ends[MAX_SEGMENTS];
    int merged_count = 0;
    
    merged_starts[0] = removed_starts[0];
    merged_ends[0] = removed_ends[0];
    merged_count = 1;
    
    for (int i = 1; i < removed_count; ++i) {
        if (removed_starts[i] <= merged_ends[merged_count - 1] + EPSILON) {
            // 可以合并
            merged_ends[merged_count - 1] = fmaxf(merged_ends[merged_count - 1], removed_ends[i]);
        } else {
            // 新区间
            merged_starts[merged_count] = removed_starts[i];
            merged_ends[merged_count] = removed_ends[i];
            merged_count++;
        }
    }
    
    // 布尔减法：长方体 - 所有圆柱体
    // 在所有剩余片段中选最长的，避免只取第一个而留下小突刺
    float box_start = box_t[0];
    float box_end = box_t[1];
    const float MIN_SEGMENT_LENGTH = 0.1f;  // 最小线段长度阈值
    
    float longest_start = 0.0f;
    float longest_end = 0.0f;
    float longest_len = -1.0f;
    float cursor = box_start;
    
    for (int i = 0; i < merged_count; ++i) {
        float rem_start = merged_starts[i];
        float rem_end = merged_ends[i];
        
        if (rem_end < box_start + EPSILON) {
            continue;
        }
        if (rem_start > box_end - EPSILON) {
            if (cursor < box_end - EPSILON) {
                float seg_len = box_end - cursor;
                if (seg_len > longest_len) {
                    longest_len = seg_len;
                    longest_start = cursor;
                    longest_end = box_end;
                }
            }
            cursor = box_end;
            break;
        }
        
        float seg_start = cursor;
        float seg_end = fminf(rem_start, box_end);
        if (seg_end > seg_start + EPSILON) {
            float seg_len = seg_end - seg_start;
            if (seg_len > longest_len) {
                longest_len = seg_len;
                longest_start = seg_start;
                longest_end = seg_end;
            }
        }
        
        if (rem_end > cursor) {
            cursor = rem_end;
        }
        if (cursor >= box_end - EPSILON) {
            break;
        }
    }
    
    if (cursor < box_end - EPSILON) {
        float seg_len = box_end - cursor;
        if (seg_len > longest_len) {
            longest_len = seg_len;
            longest_start = cursor;
            longest_end = box_end;
        }
    }
    
    if (longest_len >= MIN_SEGMENT_LENGTH) {
        float result_start = longest_start;
        float result_end = longest_end;
        validFlags[idx] = 1;
        
        if (idx < linesPerPlane) {
            // Z方向：t转换为实际z坐标
            float z_start = L.origin.z + result_start * L.direction.z;
            float z_end = L.origin.z + result_end * L.direction.z;
            visibleLines[idx].origin = Point3D(L.origin.x, L.origin.y, z_start);
            visibleLines[idx].direction = Point3D(0, 0, z_end - z_start);
            intersections[idx] = Point3D(L.origin.x, L.origin.y, (z_start + z_end) / 2.0f);
        } else if (idx < 2 * linesPerPlane) {
            // Y方向：t转换为实际y坐标
            float y_start = L.origin.y + result_start * L.direction.y;
            float y_end = L.origin.y + result_end * L.direction.y;
            visibleLines[idx].origin = Point3D(L.origin.x, y_start, L.origin.z);
            visibleLines[idx].direction = Point3D(0, y_end - y_start, 0);
            intersections[idx] = Point3D(L.origin.x, (y_start + y_end) / 2.0f, L.origin.z);
        } else {
            // X方向：t转换为实际x坐标
            float x_start = L.origin.x + result_start * L.direction.x;
            float x_end = L.origin.x + result_end * L.direction.x;
            visibleLines[idx].origin = Point3D(x_start, L.origin.y, L.origin.z);
            visibleLines[idx].direction = Point3D(x_end - x_start, 0, 0);
            intersections[idx] = Point3D((x_start + x_end) / 2.0f, L.origin.y, L.origin.z);
        }
    }
}

// 主机端包装函数（支持多圆柱体路径）
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
    GPUPerformanceMetrics* metrics
) {
    int numLines = lines.size();
    int numCylinders = cylinderPositions.size();
    
    // 如果没有圆柱体路径，至少分配1个位置（但设置为无效位置）
    int numCylindersToAllocate = (numCylinders > 0) ? numCylinders : 1;
    
    // 创建CUDA事件用于精确计时
    cudaEvent_t start, stop, h2d_start, h2d_stop, kernel_start, kernel_stop, d2h_start, d2h_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);
    
    cudaEventRecord(start, 0);
    
    // 分配设备内存
    Line* d_lines;
    Point3D* d_intersections;
    int* d_validFlags;
    Line* d_visibleLines;
    CylinderPosition* d_cylinderPositions;
    
    size_t memoryUsed = numLines * (sizeof(Line) * 2 + sizeof(Point3D) + sizeof(int)) 
                       + numCylinders * sizeof(CylinderPosition);
    
    cudaMalloc(&d_lines, numLines * sizeof(Line));
    cudaMalloc(&d_intersections, numLines * sizeof(Point3D));
    cudaMalloc(&d_validFlags, numLines * sizeof(int));
    cudaMalloc(&d_visibleLines, numLines * sizeof(Line));
    cudaMalloc(&d_cylinderPositions, numCylindersToAllocate * sizeof(CylinderPosition));
    
    // 复制输入数据到设备
    cudaEventRecord(h2d_start, 0);
    cudaMemcpy(d_lines, lines.data(), numLines * sizeof(Line), cudaMemcpyHostToDevice);
    if (numCylinders > 0) {
        cudaMemcpy(d_cylinderPositions, cylinderPositions.data(), numCylinders * sizeof(CylinderPosition), cudaMemcpyHostToDevice);
    }
    cudaEventRecord(h2d_stop, 0);
    
    // 配置kernel启动参数
    int blockSize = 256;
    int gridSize = (numLines + blockSize - 1) / blockSize;
    
    // 启动kernel
    cudaEventRecord(kernel_start, 0);
    computeBooleanSubtractionKernel<<<gridSize, blockSize>>>(
        d_lines, d_intersections, d_validFlags, d_visibleLines,
        numLines, linesPerPlane,
        cylinderRadius, cylinderHeight,
        d_cylinderPositions, numCylinders,
        boxWidth, boxDepth, boxHeight
    );
    cudaEventRecord(kernel_stop, 0);
    
    cudaDeviceSynchronize();
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
    }
    
    // 复制结果回主机
    cudaEventRecord(d2h_start, 0);
    cudaMemcpy(intersections.data(), d_intersections, numLines * sizeof(Point3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(validFlags.data(), d_validFlags, numLines * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(visibleLines.data(), d_visibleLines, numLines * sizeof(Line), cudaMemcpyDeviceToHost);
    cudaEventRecord(d2h_stop, 0);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // 释放设备内存
    cudaFree(d_lines);
    cudaFree(d_intersections);
    cudaFree(d_validFlags);
    cudaFree(d_visibleLines);
    cudaFree(d_cylinderPositions);
    
    // 计算性能指标
    if (metrics) {
        cudaEventElapsedTime(&metrics->totalTime, start, stop);
        cudaEventElapsedTime(&metrics->memcpyH2D, h2d_start, h2d_stop);
        cudaEventElapsedTime(&metrics->kernelExecution, kernel_start, kernel_stop);
        cudaEventElapsedTime(&metrics->memcpyD2H, d2h_start, d2h_stop);
        
        metrics->processedRays = numLines;
        metrics->validIntersections = 0;
        for (int i = 0; i < numLines; ++i) {
            if (validFlags[i]) metrics->validIntersections++;
        }
        metrics->blockSize = blockSize;
        metrics->gridSize = gridSize;
        metrics->deviceMemoryUsed = memoryUsed;
    }
    
    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);
}

// ============================================
// GPU数据持久化优化版本（避免每帧重新分配和传输）
// ============================================

// 全局GPU数据指针（持久化在显存中）
static Line* g_d_lines = nullptr;
static Point3D* g_d_intersections = nullptr;
static int* g_d_validFlags = nullptr;
static Line* g_d_visibleLines = nullptr;
static int g_numLines = 0;

// 初始化GPU数据（在程序开始时调用一次）
void initializeGPUData(const vector<Line>& lines) {
    // 如果已经初始化，先清理
    if (g_d_lines != nullptr) {
        cleanupGPUData();
    }
    
    g_numLines = lines.size();
    
    // 分配GPU显存（只分配一次）
    cudaMalloc(&g_d_lines, g_numLines * sizeof(Line));
    cudaMalloc(&g_d_intersections, g_numLines * sizeof(Point3D));
    cudaMalloc(&g_d_validFlags, g_numLines * sizeof(int));
    cudaMalloc(&g_d_visibleLines, g_numLines * sizeof(Line));
    
    // 传输tri-dexel线段数据（只传输一次，因为不会改变）
    cudaMemcpy(g_d_lines, lines.data(), g_numLines * sizeof(Line), cudaMemcpyHostToDevice);
    
    cout << "[GPU优化] 数据已加载到显存: " << g_numLines << " 条线段, "
         << (g_numLines * sizeof(Line) / 1024.0f / 1024.0f) << " MB" << endl;
}

// 清理GPU数据（在程序结束时调用）
void cleanupGPUData() {
    if (g_d_lines) cudaFree(g_d_lines);
    if (g_d_intersections) cudaFree(g_d_intersections);
    if (g_d_validFlags) cudaFree(g_d_validFlags);
    if (g_d_visibleLines) cudaFree(g_d_visibleLines);
    
    g_d_lines = nullptr;
    g_d_intersections = nullptr;
    g_d_validFlags = nullptr;
    g_d_visibleLines = nullptr;
    g_numLines = 0;
}

// 优化版本：只传输动态数据（铣刀路径），静态数据常驻GPU
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
    GPUPerformanceMetrics* metrics
) {
    if (g_d_lines == nullptr) {
        cerr << "[错误] GPU数据未初始化，请先调用 initializeGPUData()" << endl;
        return;
    }
    
    int numCylinders = cylinderPositions.size();
    int numCylindersToAllocate = (numCylinders > 0) ? numCylinders : 1;
    
    // 创建CUDA事件
    cudaEvent_t start, stop, h2d_start, h2d_stop, kernel_start, kernel_stop, d2h_start, d2h_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);
    
    cudaEventRecord(start, 0);
    
    // 只分配和传输动态数据（铣刀路径）
    CylinderPosition* d_cylinderPositions;
    cudaMalloc(&d_cylinderPositions, numCylindersToAllocate * sizeof(CylinderPosition));
    
    cudaEventRecord(h2d_start, 0);
    if (numCylinders > 0) {
        cudaMemcpy(d_cylinderPositions, cylinderPositions.data(), 
                   numCylinders * sizeof(CylinderPosition), cudaMemcpyHostToDevice);
    }
    cudaEventRecord(h2d_stop, 0);
    
    // 配置kernel
    int blockSize = 256;
    int gridSize = (g_numLines + blockSize - 1) / blockSize;
    
    // 启动kernel（使用持久化的GPU数据）
    cudaEventRecord(kernel_start, 0);
    computeBooleanSubtractionKernel<<<gridSize, blockSize>>>(
        g_d_lines, g_d_intersections, g_d_validFlags, g_d_visibleLines,
        g_numLines, linesPerPlane,
        cylinderRadius, cylinderHeight,
        d_cylinderPositions, numCylinders,
        boxWidth, boxDepth, boxHeight
    );
    cudaEventRecord(kernel_stop, 0);
    
    cudaDeviceSynchronize();
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
    }
    
    // 只传回结果数据
    cudaEventRecord(d2h_start, 0);
    cudaMemcpy(intersections.data(), g_d_intersections, g_numLines * sizeof(Point3D), cudaMemcpyDeviceToHost);
    cudaMemcpy(validFlags.data(), g_d_validFlags, g_numLines * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(visibleLines.data(), g_d_visibleLines, g_numLines * sizeof(Line), cudaMemcpyDeviceToHost);
    cudaEventRecord(d2h_stop, 0);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // 只释放动态数据
    cudaFree(d_cylinderPositions);
    
    // 计算性能指标
    if (metrics) {
        cudaEventElapsedTime(&metrics->totalTime, start, stop);
        cudaEventElapsedTime(&metrics->memcpyH2D, h2d_start, h2d_stop);
        cudaEventElapsedTime(&metrics->kernelExecution, kernel_start, kernel_stop);
        cudaEventElapsedTime(&metrics->memcpyD2H, d2h_start, d2h_stop);
        
        metrics->processedRays = g_numLines;
        metrics->validIntersections = 0;
        for (int i = 0; i < g_numLines; ++i) {
            if (validFlags[i]) metrics->validIntersections++;
        }
        metrics->blockSize = blockSize;
        metrics->gridSize = gridSize;
        metrics->deviceMemoryUsed = g_numLines * (sizeof(Line) * 2 + sizeof(Point3D) + sizeof(int))
                                   + numCylinders * sizeof(CylinderPosition);
    }
    
    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);
}
