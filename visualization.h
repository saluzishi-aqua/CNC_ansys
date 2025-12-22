#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "line_surface_intersection.cuh"
#include <GL/glut.h>
#include <vector>

using namespace std;

class Visualizer {
private:
    vector<Line> lines;
    vector<Point3D> intersections;
    vector<int> validFlags;

    float cameraAngleX = 0.0f;
    float cameraAngleY = 0.0f;
    float cameraDistance = 20.0f;
    int mouseX = 0, mouseY = 0;
    bool mouseLeftDown = false;

public:
    void initialize(int numLines);
    void display();
    void reshape(int width, int height);
    void keyboard(unsigned char key, int x, int y);
    void mouse(int button, int state, int x, int y);
    void motion(int x, int y);

    void drawCoordinateSystem();
    void drawSurface();
    void drawLines();
    void drawIntersections();

    static void displayWrapper();
    static void reshapeWrapper(int width, int height);
    static void keyboardWrapper(unsigned char key, int x, int y);
    static void mouseWrapper(int button, int state, int x, int y);
    static void motionWrapper(int x, int y);

    static Visualizer* instance;
};

void initializeOpenGL(int argc, char** argv, int numLines = 1000);

#endif
