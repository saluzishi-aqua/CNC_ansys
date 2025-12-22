#define _USE_MATH_DEFINES
#include "visualization.h"
#include <iostream>
#include <random>
#include <cmath>

Visualizer* Visualizer::instance = nullptr;

void Visualizer::initialize(int numLines) {
    // 生成随机直线
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> posDist(-8.0f, 8.0f);
    uniform_real_distribution<float> dirDist(-1.0f, 1.0f);

    lines.clear();
    for (int i = 0; i < numLines; ++i) {
        Line line;
        line.origin = Point3D(posDist(gen), posDist(gen), posDist(gen));
        line.direction = Point3D(dirDist(gen), dirDist(gen), dirDist(gen));
        lines.push_back(line);
    }

    // 计算交点
    findIntersectionsGPU(lines, intersections, validFlags, 0.01f, 1000);

    cout << "初始化完成: " << numLines << " 条直线" << endl;
    int validCount = 0;
    for (int flag : validFlags) {
        if (flag) validCount++;
    }
    cout << "找到 " << validCount << " 个交点" << endl;
}

void Visualizer::drawCoordinateSystem() {
    glLineWidth(2.0f);

    // X轴 - 红色
    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(10.0f, 0.0f, 0.0f);
    glEnd();

    // Y轴 - 绿色
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 10.0f, 0.0f);
    glEnd();

    // Z轴 - 蓝色
    glColor3f(0.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 10.0f);
    glEnd();

    glLineWidth(1.0f);
}

void Visualizer::drawSurface() {
    // 绘制椭球面网格
    glColor3f(0.3f, 0.3f, 0.8f);
    glBegin(GL_LINES);

    const int segments = 30;
    const float a = 4.0f, b = 3.0f, c = 4.0f; // 椭球参数

    // 经线
    for (int i = 0; i < segments; i++) {
        float theta1 = 2.0f * M_PI * i / segments;
        float theta2 = 2.0f * M_PI * (i + 1) / segments;

        for (int j = 0; j < segments; j++) {
            float phi1 = M_PI * j / segments;
            float phi2 = M_PI * (j + 1) / segments;

            float x1 = a * sin(phi1) * cos(theta1);
            float y1 = b * sin(phi1) * sin(theta1);
            float z1 = c * cos(phi1);

            float x2 = a * sin(phi1) * cos(theta2);
            float y2 = b * sin(phi1) * sin(theta2);
            float z2 = c * cos(phi1);

            glVertex3f(x1, y1, z1);
            glVertex3f(x2, y2, z2);
        }
    }

    // 纬线
    for (int i = 0; i < segments; i++) {
        float phi1 = M_PI * i / segments;
        float phi2 = M_PI * (i + 1) / segments;

        for (int j = 0; j < segments; j++) {
            float theta1 = 2.0f * M_PI * j / segments;
            float theta2 = 2.0f * M_PI * (j + 1) / segments;

            float x1 = a * sin(phi1) * cos(theta1);
            float y1 = b * sin(phi1) * sin(theta1);
            float z1 = c * cos(phi1);

            float x2 = a * sin(phi2) * cos(theta1);
            float y2 = b * sin(phi2) * sin(theta1);
            float z2 = c * cos(phi2);

            glVertex3f(x1, y1, z1);
            glVertex3f(x2, y2, z2);
        }
    }

    glEnd();
}

void Visualizer::drawLines() {
    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);

    for (int i = 0; i < lines.size(); i++) {
        const Line& line = lines[i];

        // 计算直线端点（延长到可视范围）
        Point3D start, end;
        start.x = line.origin.x - line.direction.x * 10.0f;
        start.y = line.origin.y - line.direction.y * 10.0f;
        start.z = line.origin.z - line.direction.z * 10.0f;

        end.x = line.origin.x + line.direction.x * 10.0f;
        end.y = line.origin.y + line.direction.y * 10.0f;
        end.z = line.origin.z + line.direction.z * 10.0f;

        glVertex3f(start.x, start.y, start.z);
        glVertex3f(end.x, end.y, end.z);
    }

    glEnd();
}

void Visualizer::drawIntersections() {
    glPointSize(5.0f);
    glBegin(GL_POINTS);

    for (int i = 0; i < intersections.size(); i++) {
        if (validFlags[i]) {
            const Point3D& p = intersections[i];
            // 有效交点用红色显示
            glColor3f(1.0f, 0.0f, 0.0f);
            glVertex3f(p.x, p.y, p.z);
        }
    }

    glEnd();
    glPointSize(1.0f);
}

void Visualizer::display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    // 设置相机
    gluLookAt(cameraDistance, cameraDistance, cameraDistance,
        0.0, 0.0, 0.0,
        0.0, 1.0, 0.0);

    glRotatef(cameraAngleX, 1.0f, 0.0f, 0.0f);
    glRotatef(cameraAngleY, 0.0f, 1.0f, 0.0f);

    // 绘制各个组件
    drawCoordinateSystem();
    drawSurface();
    drawLines();
    drawIntersections();

    glutSwapBuffers();
}

void Visualizer::reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (float)width / height, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
}

void Visualizer::keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 27: // ESC键
        exit(0);
        break;
    case 'r':
    case 'R':
        cameraAngleX = 0.0f;
        cameraAngleY = 0.0f;
        cameraDistance = 20.0f;
        break;
    case 'w':
    case 'W':
        cameraDistance -= 1.0f;
        break;
    case 's':
    case 'S':
        cameraDistance += 1.0f;
        break;
    }
    glutPostRedisplay();
}

void Visualizer::mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON) {
        if (state == GLUT_DOWN) {
            mouseLeftDown = true;
            mouseX = x;
            mouseY = y;
        }
        else {
            mouseLeftDown = false;
        }
    }
}

void Visualizer::motion(int x, int y) {
    if (mouseLeftDown) {
        cameraAngleY += (x - mouseX);
        cameraAngleX += (y - mouseY);
        mouseX = x;
        mouseY = y;
    }
    glutPostRedisplay();
}

// 静态包装函数
void Visualizer::displayWrapper() {
    if (instance) instance->display();
}

void Visualizer::reshapeWrapper(int width, int height) {
    if (instance) instance->reshape(width, height);
}

void Visualizer::keyboardWrapper(unsigned char key, int x, int y) {
    if (instance) instance->keyboard(key, x, y);
}

void Visualizer::mouseWrapper(int button, int state, int x, int y) {
    if (instance) instance->mouse(button, state, x, y);
}

void Visualizer::motionWrapper(int x, int y) {
    if (instance) instance->motion(x, y);
}

//void initializeOpenGL(int argc, char** argv, int numLines) {
//    glutInit(&argc, argv);
//    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
//    glutInitWindowSize(800, 600);
//    glutCreateWindow("直线与曲面相交可视化 - GPU加速");
//
//    glEnable(GL_DEPTH_TEST);
//    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
//
//    Visualizer::instance = new Visualizer();
//    Visualizer::instance->initialize(numLines);
//
//    glutDisplayFunc(Visualizer::displayWrapper);
//    glutReshapeFunc(Visualizer::reshapeWrapper);
//    glutKeyboardFunc(Visualizer::keyboardWrapper);
//    glutMouseFunc(Visualizer::mouseWrapper);
//    glutMotionFunc(Visualizer::motionWrapper);
//
//    cout << "控制说明:" << endl;
//    cout << "- 鼠标拖拽: 旋转视角" << endl;
//    cout << "- W/S: 缩放" << endl;
//    cout << "- R: 重置视角" << endl;
//    cout << "- ESC: 退出" << endl;
//
//    glutMainLoop();
//}