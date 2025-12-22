#define _USE_MATH_DEFINES
#include <GL/glut.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cmath>
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
    
    // 对三个轴向的平面进行求交
    // X方向的两个面
    if (fabs(line.direction.x) > 1e-6f) {
        float t1 = (-BOX_WIDTH/2.0f - line.origin.x) / line.direction.x;
        float t2 = (BOX_WIDTH/2.0f - line.origin.x) / line.direction.x;
        t_values.push_back(t1);
        t_values.push_back(t2);
    }
    
    // Y方向的两个面
    if (fabs(line.direction.y) > 1e-6f) {
        float t1 = (-BOX_DEPTH/2.0f - line.origin.y) / line.direction.y;
        float t2 = (BOX_DEPTH/2.0f - line.origin.y) / line.direction.y;
        t_values.push_back(t1);
        t_values.push_back(t2);
    }
    
    // Z方向的两个面
    if (fabs(line.direction.z) > 1e-6f) {
        float t1 = (-BOX_HEIGHT/2.0f - line.origin.z) / line.direction.z;
        float t2 = (BOX_HEIGHT/2.0f - line.origin.z) / line.direction.z;
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

// 布尔减法运算：计算圆柱体 - 长方体的Tri-Dexel模型（调用GPU版本）
void computeBooleanSubtraction() {
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    
    intersections.resize(lines.size());
    validFlags.resize(lines.size());
    visibleLines.resize(lines.size());
    
    // 调用GPU并行计算
    computeBooleanSubtractionGPU(
        lines,
        intersections,
        validFlags,
        visibleLines,
        linesPerPlane,
        CYLINDER_RADIUS,
        CYLINDER_HEIGHT,
        BOX_WIDTH,
        BOX_DEPTH,
        BOX_HEIGHT
    );
}

// CPU版本的布尔减法运算（用于性能对比）
void computeBooleanSubtractionCPU() {
    int gridPoints = static_cast<int>(PLANE_SIZE / GRID_SPACING) + 1;
    int linesPerPlane = gridPoints * gridPoints;
    int total = lines.size();

    // 临时变量，不影响实际结果
    vector<Point3D> temp_intersections(total, Point3D(0,0,0));
    vector<int> temp_validFlags(total, 0);
    vector<Line> temp_visibleLines(total, Line{Point3D(0,0,0), Point3D(0,0,0)});

    for (int i = 0; i < total; ++i) {
        const Line& L = lines[i];
        
        // 计算与长方体的交线段
        vector<float> box_t = lineBoxIntersection(L);
        
        // 根据方向计算与圆柱体的交线段
        vector<Segment> cylinder_segs;
        vector<Segment> box_segs;
        
        if (i < linesPerPlane) {
            // Z 方向线：x,y 常量
            float x = L.origin.x;
            float y = L.origin.y;
            if (x*x + y*y <= CYLINDER_RADIUS*CYLINDER_RADIUS) {
                float cyl_z_start = -CYLINDER_HEIGHT/2.0f;
                float cyl_z_end = CYLINDER_HEIGHT/2.0f;
                cylinder_segs.push_back(Segment(cyl_z_start, cyl_z_end));
            }
            
            if (box_t.size() >= 2) {
                box_segs.push_back(Segment(box_t[0], box_t[1]));
            }
        } else if (i < 2*linesPerPlane) {
            // Y 方向线
            float x = L.origin.x;
            float z = L.origin.z;
            if (x*x <= CYLINDER_RADIUS*CYLINDER_RADIUS && fabs(z) <= CYLINDER_HEIGHT/2.0f) {
                float delta = sqrt(CYLINDER_RADIUS*CYLINDER_RADIUS - x*x);
                cylinder_segs.push_back(Segment(-delta, delta));
            }
            
            if (box_t.size() >= 2) {
                box_segs.push_back(Segment(box_t[0], box_t[1]));
            }
        } else {
            // X 方向线
            float y = L.origin.y;
            float z = L.origin.z;
            if (y*y <= CYLINDER_RADIUS*CYLINDER_RADIUS && fabs(z) <= CYLINDER_HEIGHT/2.0f) {
                float delta = sqrt(CYLINDER_RADIUS*CYLINDER_RADIUS - y*y);
                cylinder_segs.push_back(Segment(-delta, delta));
            }
            
            if (box_t.size() >= 2) {
                box_segs.push_back(Segment(box_t[0], box_t[1]));
            }
        }
        
        // 布尔减法：圆柱体 - 长方体
        vector<Segment> result_segs;
        for (const Segment& c : cylinder_segs) {
            float start = c.t_start;
            float end = c.t_end;
            
            for (const Segment& b : box_segs) {
                if (b.t_start <= start && b.t_end >= end) {
                    start = end;
                    break;
                } else if (b.t_start > start && b.t_start < end) {
                    if (b.t_end < end) {
                        result_segs.push_back(Segment(start, b.t_start));
                        start = b.t_end;
                    } else {
                        end = b.t_start;
                    }
                } else if (b.t_end > start && b.t_end < end) {
                    start = b.t_end;
                }
            }
            
            if (end > start) {
                result_segs.push_back(Segment(start, end));
            }
        }
        
        if (!result_segs.empty()) {
            temp_validFlags[i] = 1;
            const Segment& seg = result_segs[0];
            float t_mid = (seg.t_start + seg.t_end) / 2.0f;
            
            if (i < linesPerPlane) {
                temp_visibleLines[i].origin = Point3D(L.origin.x, L.origin.y, seg.t_start);
                temp_visibleLines[i].direction = Point3D(0, 0, seg.t_end - seg.t_start);
                temp_intersections[i] = Point3D(L.origin.x, L.origin.y, t_mid);
            } else if (i < 2*linesPerPlane) {
                temp_visibleLines[i].origin = Point3D(L.origin.x, seg.t_start, L.origin.z);
                temp_visibleLines[i].direction = Point3D(0, seg.t_end - seg.t_start, 0);
                temp_intersections[i] = Point3D(L.origin.x, t_mid, L.origin.z);
            } else {
                temp_visibleLines[i].origin = Point3D(seg.t_start, L.origin.y, L.origin.z);
                temp_visibleLines[i].direction = Point3D(seg.t_end - seg.t_start, 0, 0);
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

    // 第一行：几何体信息
    string info = "圆柱体: 半径=" + to_string(CYLINDER_RADIUS) + ", 高度=" + to_string(CYLINDER_HEIGHT)
        + " | 交点: Z:" + to_string(zCnt) + " Y:" + to_string(yCnt) + " X:" + to_string(xCnt)
        + " | 显示线段:" + to_string(count(validFlags.begin(), validFlags.end(), 1));
    glRasterPos2f(10, 70);
    for (char c : info) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);

    // 第二行：GPU性能信息（黄色高亮）
    glColor3f(1.0f, 1.0f, 0.0f);
    char perfInfo[256];
    sprintf_s(perfInfo, "GPU加速: CPU耗时=%.2fms | GPU耗时=%.2fms | 加速比=%.2fx (GPU比CPU快%.1f倍)", 
              cpuTime, gpuTime, speedup, speedup);
    glRasterPos2f(10, 50);
    for (int i = 0; perfInfo[i] != '\0'; ++i) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, perfInfo[i]);
    }

    // 第三行：控制说明
    glColor3f(1,1,1);
    string controls = "控制: 鼠标旋转, W/S缩放, R重置, P打印坐标, ESC退出";
    glRasterPos2f(10, 30);
    for (char c : controls) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    
    // 第四行：数据规模
    glColor3f(0.7f, 0.7f, 0.7f);
    string dataInfo = "测试规模: " + to_string(lines.size()) + " 条射线, " 
                    + to_string(gridPoints) + "x" + to_string(gridPoints) + " 网格";
    glRasterPos2f(10, 10);
    for (char c : dataInfo) glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);

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
    drawBox();  // 绘制长方体
    
    // 绘制长方体的Tri-Dexel线段
    drawBoxTriDexel();

    // 绘制可见线段（按索引分组着色）
    drawVisibleLineSegments();
    
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
    case 27: exit(0); break;
    case 'r': case 'R':
        cameraAngleX = 30.0f; cameraAngleY = 45.0f; cameraDistance = 25.0f; break;
    case 'w': case 'W': cameraDistance -= 1.0f; break;
    case 's': case 'S': cameraDistance += 1.0f; break;
    case 'p': case 'P': printIntersectionCoordinates(); break;
    }
    glutPostRedisplay();
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
    
    cout << "\n========== 性能对比测试 ==========" << endl;
    cout << "测试数据量: " << totalLines << " 条射线" << endl;
    
    // ===== CPU版本测试 =====
    cout << "\n[CPU计算] 开始..." << endl;
    auto cpuStart = chrono::high_resolution_clock::now();
    computeBooleanSubtractionCPU();
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