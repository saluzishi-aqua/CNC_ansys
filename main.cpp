#define _USE_MATH_DEFINES
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cmath>
#include <ctime>
#include <algorithm>
#include "line_surface_intersection.cuh"

using namespace std;

// 全局变量
vector<Line> lines;
vector<Point3D> intersections;
vector<int> validFlags;
vector<Line> visibleLines; // 与 lines 对齐（相同大小），只有 validFlags[i]==1 时该条线段有效

// 性能统计变量
float gpuTime = 0.0f;  // GPU计算时间（毫秒）
float cpuTime = 0.0f;  // CPU计算时间（毫秒）
float speedup = 0.0f;  // 加速比
GPUPerformanceMetrics gpuMetrics;  // GPU详细指标

// 动画系统变量
bool animationEnabled = true;      // 是否启用动画
float animationTime = 0.0f;        // 动画时间（秒）
float cylinderPosX = 6.0f;         // 圆柱体中心X位置（铣刀与工件圆柱体外切：3.0+3.0=6.0）
float cylinderPosY = 0.0f;         // 圆柱体中心Y位置
float cylinderPosZ = 0.0f;         // 圆柱体中心Z位置
const float ANIMATION_SPEED = 2.0f;  // 动画速度（提升到2.0加快切削）
const float ANIMATION_RADIUS = 6.0f; // 圆柱体运动轨迹半径（工件半径3.0 + 铣刀半径3.0 = 6.0，铣刀外切于黄色交线）

// 铣削路径历史记录（用于累积材料移除）
vector<CylinderPosition> millingPath;  // 记录所有经过的圆柱体位置
const int MAX_PATH_POINTS = 500;       // 限制路径点数，平衡性能和效果（500点约1.3圈，保留更长切削历史）
const float PATH_SAMPLE_DISTANCE = 0.02f; // 路径采样距离（增大到0.02加快速度，减少路径点）

// 周期计数
int cycleCount = 0;                    // 已完成周期数
float lastAngle = 0.0f;                // 上一帧角度（用于检测过零点）

// 相机控制变量
float cameraAngleX = 30.0f;
float cameraAngleY = 45.0f;
float cameraDistance = 25.0f;
int mouseX = 0, mouseY = 0;
bool mouseLeftDown = false;
int windowWidth = 800;
int windowHeight = 600;

// 圆柱体参数
const float CYLINDER_RADIUS = 3.0f;
const float CYLINDER_HEIGHT = 10.0f;

// 长方体参数
const float BOX_WIDTH = 8.0f;   // X方向宽度
const float BOX_DEPTH = 8.0f;   // Y方向深度
const float BOX_HEIGHT = 8.0f;  // Z方向高度

// 平面参数（增大网格以展示GPU优势）
const float PLANE_SIZE = 50.0f;  // 从10改为50，增加5倍
const float GRID_SPACING = 0.5f;  // 从1改为0.5，网格密度提高2倍
// 总射线数：101×101×3 = 30,603 条（足以展示GPU优势）

// 检查点是否在圆柱体内部
bool isPointInsideCylinder(const Point3D& p) {
    bool insideRadius = (p.x * p.x + p.y * p.y) <= (CYLINDER_RADIUS * CYLINDER_RADIUS);
    bool insideHeight = (p.z >= -CYLINDER_HEIGHT / 2) && (p.z <= CYLINDER_HEIGHT / 2);
    return insideRadius && insideHeight;
}

// 检查点是否在长方体内部
bool isPointInsideBox(const Point3D& p) {
    bool insideX = (fabs(p.x) <= BOX_WIDTH / 2.0f);
    bool insideY = (fabs(p.y) <= BOX_DEPTH / 2.0f);
    bool insideZ = (fabs(p.z) <= BOX_HEIGHT / 2.0f);
    return insideX && insideY && insideZ;
}

// 结构体表示线段的起止参数t
struct Segment {
    float t_start;
    float t_end;
    Segment(float s, float e) : t_start(s), t_end(e) {}
};

// 计算直线与长方体的交点参数t（返回两个t值，如果有交点）
vector<float> lineBoxIntersection(const Line& line) {
    vector<float> t_values;
    
    // 对长方体尺寸轻微内缩，避免边界噪点
    const float BOX_SHRINK = 0.02f;
    float widthHalf = BOX_WIDTH/2.0f - BOX_SHRINK;
    float depthHalf = BOX_DEPTH/2.0f - BOX_SHRINK;
    float heightHalf = BOX_HEIGHT/2.0f - BOX_SHRINK;
    
    // 对三个轴向的平面进行求交
    // X方向的两个面
    if (fabs(line.direction.x) > 1e-6f) {
        float t1 = (-widthHalf - line.origin.x) / line.direction.x;
        float t2 = (widthHalf - line.origin.x) / line.direction.x;
        t_values.push_back(t1);
        t_values.push_back(t2);
    }
    
    // Y方向的两个面
    if (fabs(line.direction.y) > 1e-6f) {
        float t1 = (-depthHalf - line.origin.y) / line.direction.y;
        float t2 = (depthHalf - line.origin.y) / line.direction.y;
        t_values.push_back(t1);
        t_values.push_back(t2);
    }
    
    // Z方向的两个面
    if (fabs(line.direction.z) > 1e-6f) {
        float t1 = (-heightHalf - line.origin.z) / line.direction.z;
        float t2 = (heightHalf - line.origin.z) / line.direction.z;
        t_values.push_back(t1);
        t_values.push_back(t2);
    }
    
    // 过滤出真正在长方体内的t值
    vector<float> valid_t;
    for (float t : t_values) {
        Point3D p(
            line.origin.x + t * line.direction.x,
            line.origin.y + t * line.direction.y,
            line.origin.z + t * line.direction.z
        );
        if (isPointInsideBox(p)) {
            valid_t.push_back(t);
        }
    }
    
    sort(valid_t.begin(), valid_t.end());
    
    // 去重
    vector<float> result;
    for (size_t i = 0; i < valid_t.size(); ++i) {
        if (i == 0 || fabs(valid_t[i] - valid_t[i-1]) > 1e-4f) {
            result.push_back(valid_t[i]);
        }
    }
    
    return result;
}

// 布尔减法运算：计算圆柱体路径 - 长方体的Tri-Dexel模型（调用GPU优化版本）
void computeBooleanSubtraction() {
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    
    intersections.resize(lines.size());
    validFlags.resize(lines.size());
    visibleLines.resize(lines.size());
    
    // 调用GPU优化版本（tri-dexel数据已在GPU显存中，只传输铣刀路径）
    computeBooleanSubtractionGPU_Optimized(
        intersections,
        validFlags,
        visibleLines,
        linesPerPlane,
        CYLINDER_RADIUS,
        CYLINDER_HEIGHT,
        millingPath,  // 传入整个路径历史
        BOX_WIDTH,
        BOX_DEPTH,
        BOX_HEIGHT,
        &gpuMetrics
    );
}

// CPU版本的布尔减法运算（用于性能对比，现在也处理路径点）
void computeBooleanSubtractionCPU(const vector<CylinderPosition>& cylinderPositions) {
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    int total = lines.size();
    int numCylinders = cylinderPositions.size();

    // 临时变量，不影响实际结果
    vector<Point3D> temp_intersections(total, Point3D(0,0,0));
    vector<int> temp_validFlags(total, 0);
    vector<Line> temp_visibleLines(total, Line{Point3D(0,0,0), Point3D(0,0,0)});

    for (int i = 0; i < total; ++i) {
        const Line& L = lines[i];
        
        // 计算与长方体的交线段
        vector<float> box_t = lineBoxIntersection(L);
        
        // 收集所有圆柱体路径位置的交线段（类似GPU kernel）
        vector<Segment> removed_segs;  // 所有需要移除的区间
        
        // 遍历所有圆柱体位置
        for (int c = 0; c < numCylinders; ++c) {
            float cylinderCenterX = cylinderPositions[c].x;
            float cylinderCenterY = cylinderPositions[c].y;
            float cylinderCenterZ = cylinderPositions[c].z;
            
            if (i < linesPerPlane) {
                // Z 方向线：x,y 常量
                float x = L.origin.x - cylinderCenterX;
                float y = L.origin.y - cylinderCenterY;
                float radiusCheck = CYLINDER_RADIUS - 0.01f;  // 轻微收缩半径避免边缘噪点
                if (x*x + y*y < radiusCheck*radiusCheck) {  // 使用<而不是<=
                    float cyl_z_start = cylinderCenterZ - CYLINDER_HEIGHT/2.0f;
                    float cyl_z_end = cylinderCenterZ + CYLINDER_HEIGHT/2.0f;
                    removed_segs.push_back(Segment(cyl_z_start - L.origin.z, cyl_z_end - L.origin.z));
                }
            } else if (i < 2*linesPerPlane) {
                // Y 方向线
                float x = L.origin.x - cylinderCenterX;
                float z = L.origin.z - cylinderCenterZ;
                float radiusCheck = CYLINDER_RADIUS - 0.01f;  // 轻微收缩半径
                if (x*x < radiusCheck*radiusCheck && fabs(z) < CYLINDER_HEIGHT/2.0f) {  // 使用<
                    float delta = sqrt(radiusCheck*radiusCheck - x*x);
                    removed_segs.push_back(Segment(cylinderCenterY - delta - L.origin.y, 
                                                   cylinderCenterY + delta - L.origin.y));
                }
            } else {
                // X 方向线
                float y = L.origin.y - cylinderCenterY;
                float z = L.origin.z - cylinderCenterZ;
                float radiusCheck = CYLINDER_RADIUS - 0.01f;  // 轻微收缩半径
                if (y*y < radiusCheck*radiusCheck && fabs(z) < CYLINDER_HEIGHT/2.0f) {  // 使用<
                    float delta = sqrt(radiusCheck*radiusCheck - y*y);
                    removed_segs.push_back(Segment(cylinderCenterX - delta - L.origin.x, 
                                                   cylinderCenterX + delta - L.origin.x));
                }
            }
        }
        
        // 长方体线段
        vector<Segment> box_segs;
        
        if (box_t.size() >= 2) {
            box_segs.push_back(Segment(box_t[0], box_t[1]));
        }
        
        // 如果没有长方体交集或没有移除区间，跳过
        if (box_segs.empty()) continue;
        
        // 排序并合并移除区间（添加epsilon容差处理浮点精度）
        const float EPSILON = 0.0001f;  // 浮点精度容差
        sort(removed_segs.begin(), removed_segs.end(), 
             [](const Segment& a, const Segment& b) { return a.t_start < b.t_start; });
        
        vector<Segment> merged_removed;
        for (const auto& seg : removed_segs) {
            if (merged_removed.empty() || seg.t_start > merged_removed.back().t_end + EPSILON) {
                merged_removed.push_back(seg);
            } else {
                merged_removed.back().t_end = max(merged_removed.back().t_end, seg.t_end);
            }
        }
        
        // 布尔减法：长方体 - 所有圆柱体（保留最长剩余片段，避免小突刺）
        float box_start = box_segs[0].t_start;
        float box_end = box_segs[0].t_end;
        const float MIN_SEGMENT_LENGTH = 0.1f;  // 最小线段长度阈值
        
        float longest_start = 0.0f;
        float longest_end = 0.0f;
        float longest_len = -1.0f;
        float cursor = box_start;
        
        for (const auto& rem : merged_removed) {
            if (rem.t_end < box_start + EPSILON) {
                continue;
            }
            if (rem.t_start > box_end - EPSILON) {
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
            float seg_end = min(rem.t_start, box_end);
            if (seg_end > seg_start + EPSILON) {
                float seg_len = seg_end - seg_start;
                if (seg_len > longest_len) {
                    longest_len = seg_len;
                    longest_start = seg_start;
                    longest_end = seg_end;
                }
            }
            
            cursor = max(cursor, rem.t_end);
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
            temp_validFlags[i] = 1;
            float t_mid = (longest_start + longest_end) / 2.0f;
            
            if (i < linesPerPlane) {
                temp_visibleLines[i].origin = Point3D(L.origin.x, L.origin.y, longest_start);
                temp_visibleLines[i].direction = Point3D(0, 0, longest_end - longest_start);
                temp_intersections[i] = Point3D(L.origin.x, L.origin.y, t_mid);
            } else if (i < 2*linesPerPlane) {
                temp_visibleLines[i].origin = Point3D(L.origin.x, longest_start, L.origin.z);
                temp_visibleLines[i].direction = Point3D(0, longest_end - longest_start, 0);
                temp_intersections[i] = Point3D(L.origin.x, t_mid, L.origin.z);
            } else {
                temp_visibleLines[i].origin = Point3D(longest_start, L.origin.y, L.origin.z);
                temp_visibleLines[i].direction = Point3D(longest_end - longest_start, 0, 0);
                temp_intersections[i] = Point3D(t_mid, L.origin.y, L.origin.z);
            }
        }
    }
}

// 生成等间距网格直线（每组 gridPoints × gridPoints，顺序：先 Z 方向，再 Y，再 X）
vector<Line> generateGridLines() {
    vector<Line> lines;

    float halfSize = PLANE_SIZE / 2.0f;
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;

    cout << "生成 " << gridPoints << "×" << gridPoints << " 网格，间隔 " << GRID_SPACING << " 单位" << endl;

    // Z 方向（垂直于 XOY）
    float zStart = -halfSize * 1.5f;
    for (int i = 0; i < gridPoints; ++i) {
        for (int j = 0; j < gridPoints; ++j) {
            float x = -halfSize + i * GRID_SPACING;
            float y = -halfSize + j * GRID_SPACING;
            Line line;
            line.origin = Point3D(x, y, zStart);
            line.direction = Point3D(0.0f, 0.0f, 1.0f);
            lines.push_back(line);
        }
    }

    // Y 方向（垂直于 XOZ）
    float yStart = -halfSize * 1.5f;
    for (int i = 0; i < gridPoints; ++i) {
        for (int j = 0; j < gridPoints; ++j) {
            float x = -halfSize + i * GRID_SPACING;
            float z = -halfSize + j * GRID_SPACING;
            Line line;
            line.origin = Point3D(x, yStart, z);
            line.direction = Point3D(0.0f, 1.0f, 0.0f);
            lines.push_back(line);
        }
    }

    // X 方向（垂直于 YOZ）
    float xStart = -halfSize * 1.5f;
    for (int i = 0; i < gridPoints; ++i) {
        for (int j = 0; j < gridPoints; ++j) {
            float y = -halfSize + i * GRID_SPACING;
            float z = -halfSize + j * GRID_SPACING;
            Line line;
            line.origin = Point3D(xStart, y, z);
            line.direction = Point3D(1.0f, 0.0f, 0.0f);
            lines.push_back(line);
        }
    }

    cout << "总共生成 " << lines.size() << " 条直线" << endl;
    return lines;
}

// 绘制圆柱体内部的线段（根据 validFlags 与 visibleLines 按原始索引绘制）
void drawVisibleLineSegments() {
    glLineWidth(3.0f);

    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    int total = lines.size();

    // 逐条按索引绘制，颜色按平面分组
    glBegin(GL_LINES);
    for (int i = 0; i < total; ++i) {
        if (!validFlags[i]) continue;
        const Line& seg = visibleLines[i];
        if (i < linesPerPlane) {
            glColor3f(1.0f, 0.5f, 0.0f); // Z 方向
        } else if (i < 2 * linesPerPlane) {
            glColor3f(0.0f, 1.0f, 0.5f); // Y 方向
        } else {
            glColor3f(0.5f, 0.0f, 1.0f); // X 方向
        }

        Point3D end;
        end.x = seg.origin.x + seg.direction.x;
        end.y = seg.origin.y + seg.direction.y;
        end.z = seg.origin.z + seg.direction.z;
        glVertex3f(seg.origin.x, seg.origin.y, seg.origin.z);
        glVertex3f(end.x, end.y, end.z);
    }
    glEnd();

    glLineWidth(1.0f);
}

// 绘制圆柱体和长方体的交界线
void drawIntersectionBoundary() {
    vector<Point3D> boundaryPoints;
    
    // 从每条Tri-Dexel线段中提取交界点
    // 交界点是：线段从圆柱体内切换到圆柱体外的位置（且同时在长方体边界上）
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    
    for (int i = 0; i < lines.size(); ++i) {
        if (!validFlags[i]) continue;
        
        const Line& originalLine = lines[i];
        const Line& visibleLine = visibleLines[i];
        
        // 计算原始线段与圆柱体和长方体的所有交点
        vector<pair<float, int>> events; // (参数t, 类型：1=进入圆柱, -1=离开圆柱, 2=进入长方体, -2=离开长方体)
        
        // 圆柱体交点
        float a = originalLine.direction.x * originalLine.direction.x + 
                  originalLine.direction.y * originalLine.direction.y;
        float b = 2.0f * (originalLine.origin.x * originalLine.direction.x + 
                         originalLine.origin.y * originalLine.direction.y);
        float c = originalLine.origin.x * originalLine.origin.x + 
                  originalLine.origin.y * originalLine.origin.y - 
                  CYLINDER_RADIUS * CYLINDER_RADIUS;
        
        if (a > 1e-6f) {
            float discriminant = b * b - 4.0f * a * c;
            if (discriminant >= 0) {
                float t1 = (-b - sqrt(discriminant)) / (2.0f * a);
                float t2 = (-b + sqrt(discriminant)) / (2.0f * a);
                
                float z1 = originalLine.origin.z + t1 * originalLine.direction.z;
                float z2 = originalLine.origin.z + t2 * originalLine.direction.z;
                
                if (fabs(z1) <= CYLINDER_HEIGHT/2.0f) events.push_back({t1, 1});
                if (fabs(z2) <= CYLINDER_HEIGHT/2.0f) events.push_back({t2, -1});
            }
        }
        
        // 长方体6个面的交点
        float tVals[6];
        int validCount = 0;
        
        // X面
        if (fabs(originalLine.direction.x) > 1e-6f) {
            tVals[validCount++] = (-BOX_WIDTH/2.0f - originalLine.origin.x) / originalLine.direction.x;
            tVals[validCount++] = (BOX_WIDTH/2.0f - originalLine.origin.x) / originalLine.direction.x;
        }
        // Y面
        if (fabs(originalLine.direction.y) > 1e-6f) {
            tVals[validCount++] = (-BOX_DEPTH/2.0f - originalLine.origin.y) / originalLine.direction.y;
            tVals[validCount++] = (BOX_DEPTH/2.0f - originalLine.origin.y) / originalLine.direction.y;
        }
        // Z面
        if (fabs(originalLine.direction.z) > 1e-6f) {
            tVals[validCount++] = (-BOX_HEIGHT/2.0f - originalLine.origin.z) / originalLine.direction.z;
            tVals[validCount++] = (BOX_HEIGHT/2.0f - originalLine.origin.z) / originalLine.direction.z;
        }
        
        // 检查哪些t值对应进入/离开长方体
        for (int j = 0; j < validCount; ++j) {
            float t = tVals[j];
            Point3D p;
            p.x = originalLine.origin.x + t * originalLine.direction.x;
            p.y = originalLine.origin.y + t * originalLine.direction.y;
            p.z = originalLine.origin.z + t * originalLine.direction.z;
            
            if (isPointInsideBox(p)) {
                // 检查这是进入还是离开
                Point3D pBefore;
                float eps = -0.001f;
                pBefore.x = originalLine.origin.x + (t + eps) * originalLine.direction.x;
                pBefore.y = originalLine.origin.y + (t + eps) * originalLine.direction.y;
                pBefore.z = originalLine.origin.z + (t + eps) * originalLine.direction.z;
                
                bool wasInside = isPointInsideBox(pBefore);
                events.push_back({t, wasInside ? -2 : 2});
            }
        }
        
        // 找到同时在圆柱体表面和长方体表面的点
        for (const auto& ev : events) {
            float t = ev.first;
            int type = ev.second;
            
            // 如果是圆柱体边界或长方体边界
            if (abs(type) == 1 || abs(type) == 2) {
                Point3D p;
                p.x = originalLine.origin.x + t * originalLine.direction.x;
                p.y = originalLine.origin.y + t * originalLine.direction.y;
                p.z = originalLine.origin.z + t * originalLine.direction.z;
                
                // 检查是否同时接近两个表面
                float distToCylinder = sqrt(p.x*p.x + p.y*p.y) - CYLINDER_RADIUS;
                float distToBox = min({
                    fabs(fabs(p.x) - BOX_WIDTH/2.0f),
                    fabs(fabs(p.y) - BOX_DEPTH/2.0f),
                    fabs(fabs(p.z) - BOX_HEIGHT/2.0f)
                });
                
                // 如果点接近圆柱表面且接近长方体表面
                if (fabs(distToCylinder) < 0.1f && distToBox < 0.1f) {
                    boundaryPoints.push_back(p);
                }
            }
        }
    }
    
    // 绘制交界点（使用大黄色点）
    glPointSize(8.0f);
    glColor3f(1.0f, 1.0f, 0.0f); // 黄色
    glBegin(GL_POINTS);
    for (const auto& p : boundaryPoints) {
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();
    glPointSize(1.0f);
}

// 绘制长方体的Tri-Dexel线段
void drawBoxTriDexel() {
    glLineWidth(2.0f);
    
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    float halfSize = PLANE_SIZE / 2.0f;
    
    glBegin(GL_LINES);
    
    // Z方向线段（穿过长方体）
    for (int i = 0; i < gridPoints; ++i) {
        for (int j = 0; j < gridPoints; ++j) {
            float x = -halfSize + i * GRID_SPACING;
            float y = -halfSize + j * GRID_SPACING;
            
            // 检查是否在长方体的XY投影内
            if (fabs(x) <= BOX_WIDTH/2.0f && fabs(y) <= BOX_DEPTH/2.0f) {
                glColor3f(1.0f, 0.3f, 0.3f); // 红色系，Z方向
                glVertex3f(x, y, -BOX_HEIGHT/2.0f);
                glVertex3f(x, y, BOX_HEIGHT/2.0f);
            }
        }
    }
    
    // Y方向线段（穿过长方体）
    for (int i = 0; i < gridPoints; ++i) {
        for (int j = 0; j < gridPoints; ++j) {
            float x = -halfSize + i * GRID_SPACING;
            float z = -halfSize + j * GRID_SPACING;
            
            // 检查是否在长方体的XZ投影内
            if (fabs(x) <= BOX_WIDTH/2.0f && fabs(z) <= BOX_HEIGHT/2.0f) {
                glColor3f(1.0f, 0.5f, 0.3f); // 橙红色，Y方向
                glVertex3f(x, -BOX_DEPTH/2.0f, z);
                glVertex3f(x, BOX_DEPTH/2.0f, z);
            }
        }
    }
    
    // X方向线段（穿过长方体）
    for (int i = 0; i < gridPoints; ++i) {
        for (int j = 0; j < gridPoints; ++j) {
            float y = -halfSize + i * GRID_SPACING;
            float z = -halfSize + j * GRID_SPACING;
            
            // 检查是否在长方体的YZ投影内
            if (fabs(y) <= BOX_DEPTH/2.0f && fabs(z) <= BOX_HEIGHT/2.0f) {
                glColor3f(1.0f, 0.7f, 0.5f); // 浅橙色，X方向
                glVertex3f(-BOX_WIDTH/2.0f, y, z);
                glVertex3f(BOX_WIDTH/2.0f, y, z);
            }
        }
    }
    
    glEnd();
    glLineWidth(1.0f);
}

void drawCoordinateSystem() {
    glLineWidth(2.0f);
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0); glVertex3f(15,0,0);
    glEnd();
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0); glVertex3f(0,15,0);
    glEnd();
    glColor3f(0.0f,0.0f,1.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0); glVertex3f(0,0,15);
    glEnd();
    glLineWidth(1.0f);
}

// 绘制毛坯红色线框（只显示边界）
void drawBoxWireframe() {
    const float w = BOX_WIDTH / 2.0f;
    const float d = BOX_DEPTH / 2.0f;
    const float h = BOX_HEIGHT / 2.0f;

    // 绘制长方体红色边框
    glColor4f(1.0f, 0.2f, 0.2f, 1.0f);  // 红色
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    // 底面四条边
    glVertex3f(-w, -d, -h); glVertex3f(w, -d, -h);
    glVertex3f(w, -d, -h); glVertex3f(w, d, -h);
    glVertex3f(w, d, -h); glVertex3f(-w, d, -h);
    glVertex3f(-w, d, -h); glVertex3f(-w, -d, -h);
    // 顶面四条边
    glVertex3f(-w, -d, h); glVertex3f(w, -d, h);
    glVertex3f(w, -d, h); glVertex3f(w, d, h);
    glVertex3f(w, d, h); glVertex3f(-w, d, h);
    glVertex3f(-w, d, h); glVertex3f(-w, -d, h);
    // 四条竖边
    glVertex3f(-w, -d, -h); glVertex3f(-w, -d, h);
    glVertex3f(w, -d, -h); glVertex3f(w, -d, h);
    glVertex3f(w, d, -h); glVertex3f(w, d, h);
    glVertex3f(-w, d, -h); glVertex3f(-w, d, h);
    glEnd();
    glLineWidth(1.0f);
}

void drawBox() {
    const float w = BOX_WIDTH / 2.0f;
    const float d = BOX_DEPTH / 2.0f;
    const float h = BOX_HEIGHT / 2.0f;

    // 绘制长方体面（半透明）
    glColor4f(0.8f, 0.3f, 0.3f, 0.4f);
    
    glBegin(GL_QUADS);
    // 前面 (Z+)
    glVertex3f(-w, -d, h); glVertex3f(w, -d, h);
    glVertex3f(w, d, h); glVertex3f(-w, d, h);
    // 后面 (Z-)
    glVertex3f(-w, -d, -h); glVertex3f(-w, d, -h);
    glVertex3f(w, d, -h); glVertex3f(w, -d, -h);
    // 顶面 (Y+)
    glVertex3f(-w, d, -h); glVertex3f(-w, d, h);
    glVertex3f(w, d, h); glVertex3f(w, d, -h);
    // 底面 (Y-)
    glVertex3f(-w, -d, -h); glVertex3f(w, -d, -h);
    glVertex3f(w, -d, h); glVertex3f(-w, -d, h);
    // 右面 (X+)
    glVertex3f(w, -d, -h); glVertex3f(w, d, -h);
    glVertex3f(w, d, h); glVertex3f(w, -d, h);
    // 左面 (X-)
    glVertex3f(-w, -d, -h); glVertex3f(-w, -d, h);
    glVertex3f(-w, d, h); glVertex3f(-w, d, -h);
    glEnd();

    // 绘制长方体边框（清晰的轮廓）
    glColor4f(0.7f, 0.1f, 0.1f, 1.0f);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // 底面四条边
    glVertex3f(-w, -d, -h); glVertex3f(w, -d, -h);
    glVertex3f(w, -d, -h); glVertex3f(w, d, -h);
    glVertex3f(w, d, -h); glVertex3f(-w, d, -h);
    glVertex3f(-w, d, -h); glVertex3f(-w, -d, -h);
    // 顶面四条边
    glVertex3f(-w, -d, h); glVertex3f(w, -d, h);
    glVertex3f(w, -d, h); glVertex3f(w, d, h);
    glVertex3f(w, d, h); glVertex3f(-w, d, h);
    glVertex3f(-w, d, h); glVertex3f(-w, -d, h);
    // 四条竖边
    glVertex3f(-w, -d, -h); glVertex3f(-w, -d, h);
    glVertex3f(w, -d, -h); glVertex3f(w, -d, h);
    glVertex3f(w, d, -h); glVertex3f(w, d, h);
    glVertex3f(-w, d, -h); glVertex3f(-w, d, h);
    glEnd();
    glLineWidth(1.0f);
}

void drawCylinder() {
    const int segments = 36;
    const float height = CYLINDER_HEIGHT;
    const float radius = CYLINDER_RADIUS;

    glPushMatrix();
    glTranslatef(cylinderPosX, cylinderPosY, cylinderPosZ);  // 动态位置

    // 绘制圆柱体侧面（半透明填充）
    glColor4f(0.3f, 0.5f, 0.8f, 0.5f);
    glBegin(GL_QUADS);
    for (int i = 0; i < segments; ++i) {
        float angle1 = 2.0f * M_PI * i / segments;
        float angle2 = 2.0f * M_PI * (i + 1) / segments;
        float x1 = radius * cos(angle1), y1 = radius * sin(angle1);
        float x2 = radius * cos(angle2), y2 = radius * sin(angle2);
        glVertex3f(x1, y1, -height/2); glVertex3f(x1, y1, height/2);
        glVertex3f(x2, y2, height/2); glVertex3f(x2, y2, -height/2);
    }
    glEnd();

    // 绘制顶面和底面
    glColor4f(0.3f, 0.5f, 0.8f, 0.5f);
    // 顶面
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0, 0, height/2);
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        glVertex3f(radius * cos(angle), radius * sin(angle), height/2);
    }
    glEnd();
    // 底面
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(0, 0, -height/2);
    for (int i = 0; i <= segments; ++i) {
        float angle = 2.0f * M_PI * i / segments;
        glVertex3f(radius * cos(angle), radius * sin(angle), -height/2);
    }
    glEnd();

    // 绘制轮廓线（更清晰）
    glColor4f(0.2f, 0.4f, 0.7f, 1.0f);
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    for (int i = 0; i < segments; ++i) {
        float angle1 = 2.0f * M_PI * i / segments;
        float angle2 = 2.0f * M_PI * (i + 1) / segments;
        float x1 = radius * cos(angle1), y1 = radius * sin(angle1);
        float x2 = radius * cos(angle2), y2 = radius * sin(angle2);
        // 顶部圆
        glVertex3f(x1, y1, height/2);  glVertex3f(x2, y2, height/2);
        // 底部圆
        glVertex3f(x1, y1, -height/2); glVertex3f(x2, y2, -height/2);
    }
    // 绘制几条竖线作为侧面轮廓
    for (int i = 0; i < 8; ++i) {
        float angle = 2.0f * M_PI * i / 8;
        float x = radius * cos(angle), y = radius * sin(angle);
        glVertex3f(x, y, -height/2); glVertex3f(x, y, height/2);
    }
    glEnd();
    glLineWidth(1.0f);
    
    glPopMatrix();  // 恢复变换矩阵
}

void drawIntersections() {
    glPointSize(6.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < intersections.size(); ++i) {
        if (!validFlags[i]) continue;
        const Point3D& p = intersections[i];
        glColor3f(1.0f,0.0f,0.0f);
        glVertex3f(p.x,p.y,p.z);
    }
    glEnd();
    glPointSize(1.0f);
}

void printIntersectionCoordinates() {
    cout << "\n=== 交点坐标 ===" << endl;
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    int cnt = 0;
    for (int i = 0; i < intersections.size(); ++i) {
        if (!validFlags[i]) continue;
        string type = (i < linesPerPlane) ? "Z" : (i < 2*linesPerPlane) ? "Y" : "X";
        const Point3D& p = intersections[i];
        cout << "交点 " << ++cnt << " (" << type << "): (" << p.x << ", " << p.y << ", " << p.z << ")" << endl;
    }
    cout << "总共找到 " << cnt << " 个交点，显示 " << count(validFlags.begin(), validFlags.end(), 1) << " 条可见线段" << endl;
}

void drawText() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, windowWidth, 0, windowHeight);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(1,1,1);
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    int zCnt=0,yCnt=0,xCnt=0;
    for (int i=0;i<lines.size();++i) if (validFlags[i]) {
        if (i<linesPerPlane) ++zCnt;
        else if (i<2*linesPerPlane) ++yCnt;
        else ++xCnt;
    }

    // 第一行：周期和铣削信息（高亮显示）
    glColor3f(1.0f, 0.5f, 0.0f);  // 橙色
    char cycleInfo[256];
    sprintf_s(cycleInfo, "周期: %d | 铣削路径: %d 点 | 交点: Z:%d Y:%d X:%d | 显示线段: %d",
              cycleCount, (int)millingPath.size(), zCnt, yCnt, xCnt,
              (int)count(validFlags.begin(), validFlags.end(), 1));
    glRasterPos2f(10, 130);
    for (int i = 0; cycleInfo[i] != '\0'; ++i) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, cycleInfo[i]);
    }
    
    // 第二行：几何体信息
    glColor3f(1, 1, 1);
    char geoInfo[256];
    sprintf_s(geoInfo, "圆柱体: 半径=%.1f 高度=%.1f | 位置=(%.2f, %.2f, %.2f)",
              CYLINDER_RADIUS, CYLINDER_HEIGHT, cylinderPosX, cylinderPosY, cylinderPosZ);
    glRasterPos2f(10, 110);
    for (int i = 0; geoInfo[i] != '\0'; ++i) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, geoInfo[i]);
    }

    // 第二行：实时计算性能对比（黄色高亮）
    glColor3f(1.0f, 1.0f, 0.0f);
    char perfInfo[256];
    sprintf_s(perfInfo, "实时对比 CPU: %.2fms | GPU: %.2fms | 加速比: %.2fx | 路径点: %d", 
              cpuTime, gpuTime, speedup, (int)millingPath.size());
    glRasterPos2f(10, 90);
    for (int i = 0; perfInfo[i] != '\0'; ++i) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, perfInfo[i]);
    }
    
    // 第三行：GPU详细指标（青色）
    glColor3f(0.3f, 0.9f, 1.0f);
    char detailInfo[512];
    sprintf_s(detailInfo, "GPU详细: 内核=%.2fms | H2D=%.2fms | D2H=%.2fms | 显存=%.2fMB | 线程=%dx%d | 射线=%d | 交点=%d",
              gpuMetrics.kernelExecution, gpuMetrics.memcpyH2D, gpuMetrics.memcpyD2H,
              gpuMetrics.deviceMemoryUsed / (1024.0f * 1024.0f),
              gpuMetrics.gridSize, gpuMetrics.blockSize,
              gpuMetrics.processedRays, gpuMetrics.validIntersections);
    glRasterPos2f(10, 70);
    for (int i = 0; detailInfo[i] != '\0'; ++i) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, detailInfo[i]);
    }

    // 第四行：控制说明
    glColor3f(1,1,1);
    string controls = "控制: 鼠标旋转, W缩放, R重置, S保存STL, SPACE 暂停/播放, P打印坐标, ESC退出";
    glRasterPos2f(10, 50);
    for (char c : controls) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    
    // 第五行：数据规模
    glColor3f(0.7f, 0.7f, 0.7f);
    string dataInfo = "测试规模: " + to_string(lines.size()) + " 条射线, " 
                    + to_string(gridPoints) + "x" + to_string(gridPoints) + " 网格";
    glRasterPos2f(10, 30);
    for (char c : dataInfo) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    
    // 第六行：动画状态（绿色）
    glColor3f(0.3f, 1.0f, 0.3f);
    string animInfo = "动画状态: " + string(animationEnabled ? "播放中" : "已暂停") 
                    + " | 时间: " + to_string(animationTime) + "s";
    glRasterPos2f(10, 10);
    for (char c : animInfo) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void drawLines() {
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (const Line& line : lines) {
        glColor3f(0.8f,0.8f,0.8f);
        Point3D end;
        end.x = line.origin.x + line.direction.x *  (PLANE_SIZE * 2.0f);
        end.y = line.origin.y + line.direction.y *  (PLANE_SIZE * 2.0f);
        end.z = line.origin.z + line.direction.z *  (PLANE_SIZE * 2.0f);
        glVertex3f(line.origin.x, line.origin.y, line.origin.z);
        glVertex3f(end.x, end.y, end.z);
    }
    glEnd();
    glLineWidth(1.0f);
}

// 导出Tri-Dexel结果到STL文件
void exportToSTL(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "无法创建STL文件: " << filename << endl;
        return;
    }
    
    cout << "开始导出STL文件: " << filename << endl;
    
    // STL ASCII格式头
    file << "solid TriDexelModel\n";
    
    int triangleCount = 0;
    
    // 遍历所有有效的tri-dexel线段
    for (int i = 0; i < lines.size(); ++i) {
        if (!validFlags[i]) continue;
        
        const Line& seg = visibleLines[i];
        Point3D start = seg.origin;
        Point3D end;
        end.x = seg.origin.x + seg.direction.x;
        end.y = seg.origin.y + seg.direction.y;
        end.z = seg.origin.z + seg.direction.z;
        
        // 为每条线段创建一个细圆柱体（用三角形近似）
        const float lineRadius = 0.05f;  // 线段半径
        const int segments = 6;  // 6边形近似圆柱
        
        // 为圆柱体的每个侧面创建两个三角形
        for (int j = 0; j < segments; ++j) {
            float angle1 = 2.0f * M_PI * j / segments;
            float angle2 = 2.0f * M_PI * (j + 1) / segments;
            
            float cos1 = cosf(angle1), sin1 = sinf(angle1);
            float cos2 = cosf(angle2), sin2 = sinf(angle2);
            
            // 计算4个顶点（根据线段方向）
            Point3D v1, v2, v3, v4;
            
            // 计算垂直于线段的两个正交向量
            Point3D dir = {end.x - start.x, end.y - start.y, end.z - start.z};
            float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
            if (len < 1e-6f) continue;
            dir.x /= len; dir.y /= len; dir.z /= len;
            
            Point3D perp1, perp2;
            if (fabsf(dir.z) < 0.9f) {
                perp1 = {-dir.y, dir.x, 0};
            } else {
                perp1 = {0, -dir.z, dir.y};
            }
            float plen1 = sqrtf(perp1.x*perp1.x + perp1.y*perp1.y + perp1.z*perp1.z);
            perp1.x /= plen1; perp1.y /= plen1; perp1.z /= plen1;
            
            perp2.x = dir.y*perp1.z - dir.z*perp1.y;
            perp2.y = dir.z*perp1.x - dir.x*perp1.z;
            perp2.z = dir.x*perp1.y - dir.y*perp1.x;
            
            float offset1x = lineRadius * (cos1 * perp1.x + sin1 * perp2.x);
            float offset1y = lineRadius * (cos1 * perp1.y + sin1 * perp2.y);
            float offset1z = lineRadius * (cos1 * perp1.z + sin1 * perp2.z);
            
            float offset2x = lineRadius * (cos2 * perp1.x + sin2 * perp2.x);
            float offset2y = lineRadius * (cos2 * perp1.y + sin2 * perp2.y);
            float offset2z = lineRadius * (cos2 * perp1.z + sin2 * perp2.z);
            
            v1 = {start.x + offset1x, start.y + offset1y, start.z + offset1z};
            v2 = {end.x + offset1x, end.y + offset1y, end.z + offset1z};
            v3 = {end.x + offset2x, end.y + offset2y, end.z + offset2z};
            v4 = {start.x + offset2x, start.y + offset2y, start.z + offset2z};
            
            // 计算法向量
            Point3D edge1 = {v2.x - v1.x, v2.y - v1.y, v2.z - v1.z};
            Point3D edge2 = {v4.x - v1.x, v4.y - v1.y, v4.z - v1.z};
            Point3D normal = {
                edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x
            };
            float nlen = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
            if (nlen > 1e-6f) {
                normal.x /= nlen; normal.y /= nlen; normal.z /= nlen;
            }
            
            // 第一个三角形
            file << "  facet normal " << normal.x << " " << normal.y << " " << normal.z << "\n";
            file << "    outer loop\n";
            file << "      vertex " << v1.x << " " << v1.y << " " << v1.z << "\n";
            file << "      vertex " << v2.x << " " << v2.y << " " << v2.z << "\n";
            file << "      vertex " << v4.x << " " << v4.y << " " << v4.z << "\n";
            file << "    endloop\n";
            file << "  endfacet\n";
            triangleCount++;
            
            // 第二个三角形
            file << "  facet normal " << normal.x << " " << normal.y << " " << normal.z << "\n";
            file << "    outer loop\n";
            file << "      vertex " << v2.x << " " << v2.y << " " << v2.z << "\n";
            file << "      vertex " << v3.x << " " << v3.y << " " << v3.z << "\n";
            file << "      vertex " << v4.x << " " << v4.y << " " << v4.z << "\n";
            file << "    endloop\n";
            file << "  endfacet\n";
            triangleCount++;
        }
    }
    
    file << "endsolid TriDexelModel\n";
    file.close();
    
    cout << "STL文件导出完成! 三角形数量: " << triangleCount << endl;
    cout << "文件位置: " << filename << endl;
}

// 绘制铣削路径轨迹（用于可视化铣刀经过的所有位置）
void drawMillingPath() {
    if (millingPath.empty()) return;
    
    const int segments = 24;
    const float radius = CYLINDER_RADIUS;
    const float height = CYLINDER_HEIGHT;
    
    // 绘制每个位置的圆柱体轮廓（紫色半透明）
    glColor4f(0.8f, 0.3f, 0.8f, 0.3f);  // 紫色
    glLineWidth(1.5f);
    
    for (const auto& pos : millingPath) {
        glPushMatrix();
        glTranslatef(pos.x, pos.y, pos.z);
        
        // 绘制顶部圆周
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < segments; ++i) {
            float angle = 2.0f * M_PI * i / segments;
            glVertex3f(radius * cos(angle), radius * sin(angle), height/2);
        }
        glEnd();
        
        // 绘制底部圆周
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < segments; ++i) {
            float angle = 2.0f * M_PI * i / segments;
            glVertex3f(radius * cos(angle), radius * sin(angle), -height/2);
        }
        glEnd();
        
        glPopMatrix();
    }
    
    // 绘制路径连线（显示铣刀移动轨迹）
    glColor4f(1.0f, 0.5f, 1.0f, 0.8f);  // 亮紫色
    glLineWidth(2.0f);
    glBegin(GL_LINE_STRIP);
    for (const auto& pos : millingPath) {
        glVertex3f(pos.x, pos.y, pos.z);
    }
    glEnd();
    
    glLineWidth(1.0f);
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    gluLookAt(cameraDistance, cameraDistance, cameraDistance,
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0);

    glRotatef(cameraAngleX, 1.0f, 0.0f, 0.0f);
    glRotatef(cameraAngleY, 0.0f, 1.0f, 0.0f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    drawCoordinateSystem();
    drawCylinder();
    
    // 绘制铣削路径轨迹（显示铣刀经过的所有位置）
    // drawMillingPath();  // 已禁用刀轨显示
    
    // 先绘制内部的tri-dexel线段（切割后的结果）
    drawVisibleLineSegments();
    
    // 只绘制红色边框线（不绘制半透明面），让切削效果可见
    drawBoxWireframe();
    
    // 绘制圆柱体和长方体的交界线（黄色高亮）
    drawIntersectionBoundary();

    // 绘制所有原始直线（可注释掉）
    // drawLines();

    drawIntersections();

    glDisable(GL_BLEND);

    drawText();

    glutSwapBuffers();
}

void reshape(int width, int height) {
    windowWidth = width; windowHeight = height;
    glViewport(0,0,width,height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)width/height, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 27: 
        cout << "\n[GPU优化] 清理显存数据..." << endl;
        cleanupGPUData();
        exit(0); 
        break;
    case 'r': case 'R':
        cameraAngleX = 30.0f; cameraAngleY = 45.0f; cameraDistance = 25.0f; break;
    case 'w': case 'W': cameraDistance -= 1.0f; break;
    case 's': case 'S': 
        // 导出STL文件
        {
            // 生成带时间戳的文件名
            time_t now = time(0);
            tm ltm;
            localtime_s(&ltm, &now);
            char filename[256];
            sprintf_s(filename, "tridexel_cycle%d_%04d%02d%02d_%02d%02d%02d.stl",
                     cycleCount,
                     1900 + ltm.tm_year, 1 + ltm.tm_mon, ltm.tm_mday,
                     ltm.tm_hour, ltm.tm_min, ltm.tm_sec);
            exportToSTL(filename);
        }
        break;
    case 'p': case 'P': printIntersectionCoordinates(); break;
    case ' ':  // 空格键控制动画播放/暂停
        animationEnabled = !animationEnabled;
        cout << "动画" << (animationEnabled ? "开启" : "暂停") << endl;
        break;
    }
    glutPostRedisplay();
}

// 动画更新函数
void updateAnimation(int value) {
    if (animationEnabled) {
        // 保存上一个位置
        float lastPosX = cylinderPosX;
        float lastPosY = cylinderPosY;
        float lastPosZ = cylinderPosZ;
        
        // 更新时间
        animationTime += 0.016f;  // 约60Hz
        
        // 圆形轨迹运动（在XY平面）
        float angle = animationTime * ANIMATION_SPEED;
        
        // 检测周期完成（角度从接近2π回到0附近）
        float normalizedAngle = fmodf(angle, 2.0f * (float)M_PI);
        if (normalizedAngle < lastAngle && lastAngle > (float)M_PI) {
            cycleCount++;
            cout << "完成第 " << cycleCount << " 周期" << endl;
        }
        lastAngle = normalizedAngle;
        
        cylinderPosX = ANIMATION_RADIUS * cosf(angle);
        cylinderPosY = ANIMATION_RADIUS * sinf(angle);
        cylinderPosZ = 0.0f;
        
        // 计算移动距离
        float dx = cylinderPosX - lastPosX;
        float dy = cylinderPosY - lastPosY;
        float dz = cylinderPosZ - lastPosZ;
        float distance = sqrtf(dx*dx + dy*dy + dz*dz);
        
        // 添加到路径历史（更宽松的条件，确保路径点被添加）
        if (distance > PATH_SAMPLE_DISTANCE) {
            millingPath.push_back(CylinderPosition(cylinderPosX, cylinderPosY, cylinderPosZ));
            
            // 限制路径点数量（超过上限时只删除1个最旧的点，保持点数稳定）
            // 这样GPU计算量保持恒定，不会出现先快后慢的问题
            while (millingPath.size() > MAX_PATH_POINTS) {
                millingPath.erase(millingPath.begin());
            }
            
            // 输出调试信息（每100个点输出一次）
            if (millingPath.size() % 100 == 0) {
                cout << "路径点数: " << millingPath.size() << ", 位置: (" 
                     << cylinderPosX << ", " << cylinderPosY << ", " << cylinderPosZ << ")" << endl;
            }
            
            // 重新计算布尔运算（同时测试CPU和GPU，实时对比性能）
            // CPU版本
            auto cpuStart = chrono::high_resolution_clock::now();
            computeBooleanSubtractionCPU(millingPath);
            auto cpuEnd = chrono::high_resolution_clock::now();
            cpuTime = chrono::duration_cast<chrono::microseconds>(cpuEnd - cpuStart).count() / 1000.0f;
            
            // GPU版本（结果用于显示）
            auto gpuStart = chrono::high_resolution_clock::now();
            computeBooleanSubtraction();
            auto gpuEnd = chrono::high_resolution_clock::now();
            gpuTime = chrono::duration_cast<chrono::microseconds>(gpuEnd - gpuStart).count() / 1000.0f;
            
            // 更新加速比
            speedup = cpuTime / gpuTime;
        }
        
        glutPostRedisplay();
    }
    
    // 16ms后再次调用（约60 FPS）
    glutTimerFunc(16, updateAnimation, 0);
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) { mouseLeftDown = true; mouseX = x; mouseY = y; }
        else mouseLeftDown = false;
    }
}

void motion(int x, int y) {
    if (mouseLeftDown) {
        cameraAngleY += (x - mouseX) * 0.5f;
        cameraAngleX += (y - mouseY) * 0.5f;
        mouseX = x; mouseY = y;
    }
    glutPostRedisplay();
}

void initializeOpenGL(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800,600);
    glutCreateWindow("圆柱体与轴向网格直线相交 - GPU加速演示");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f,0.1f,0.1f,1.0f);

    lines = generateGridLines();
    int totalLines = lines.size();
    
    // 初始化GPU数据持久化（tri-dexel线段常驻显存）
    cout << "\n[GPU优化] 初始化显存数据..." << endl;
    initializeGPUData(lines);
    
    // 初始化铣削路径（为空，让铣刀开始移动后才积累）
    millingPath.clear();
    // 不添加初始位置，让初始显示为完整的长方体工件
    
    cout << "\n========== 性能对比测试 ==========" << endl;
    cout << "测试数据量: " << totalLines << " 条射线" << endl;
    
    // ===== CPU版本测试 =====
    cout << "\n[CPU计算] 开始..." << endl;
    auto cpuStart = chrono::high_resolution_clock::now();
    computeBooleanSubtractionCPU(millingPath);  // 传入空路径进行基准测试
    auto cpuEnd = chrono::high_resolution_clock::now();
    cpuTime = chrono::duration_cast<chrono::microseconds>(cpuEnd - cpuStart).count() / 1000.0f;
    cout << "[CPU计算] 完成，耗时: " << cpuTime << " ms" << endl;
    
    // ===== GPU版本测试 =====
    cout << "\n[GPU计算] 开始..." << endl;
    auto gpuStart = chrono::high_resolution_clock::now();
    computeBooleanSubtraction();
    auto gpuEnd = chrono::high_resolution_clock::now();
    gpuTime = chrono::duration_cast<chrono::microseconds>(gpuEnd - gpuStart).count() / 1000.0f;
    cout << "[GPU计算] 完成，耗时: " << gpuTime << " ms" << endl;
    
    // ===== 性能对比 =====
    speedup = cpuTime / gpuTime;
    cout << "\n========== 性能对比结果 ==========" << endl;
    cout << "CPU耗时: " << cpuTime << " ms" << endl;
    cout << "GPU耗时: " << gpuTime << " ms" << endl;
    cout << "加速比: " << speedup << "x (GPU比CPU快 " << speedup << " 倍)" << endl;
    cout << "有效交点数: " << count(validFlags.begin(), validFlags.end(), 1) << endl;
    cout << "================================\n" << endl;

    printIntersectionCoordinates();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutTimerFunc(16, updateAnimation, 0);  // 启动动画循环

    glutMainLoop();
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "使用GPU: " << prop.name << endl;
    cout << "显存: " << prop.totalGlobalMem / (1024*1024) << " MB" << endl;

    initializeOpenGL(argc, argv);
    return 0;
}
