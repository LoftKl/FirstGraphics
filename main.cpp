// main.cpp
#include <GLUT/glut.h>
#include "Quaternion.h"
#include <cmath>

// current orientation
Quaternion<float> orientation;

// how much to rotate per key press (in radians)
const float ROT_SPEED = 5.0f * (M_PI / 180.0f);

void keyboard(unsigned char key, int x, int y) {
    Quaternion<float> delta;
    switch (key) {
        case 'q':  // roll CCW around Z
            delta = Quaternion<float>::fromAxisAngle( ROT_SPEED, 0.0f, 0.0f, 1.0f);
            break;
        case 'e':  // roll CW around Z
            delta = Quaternion<float>::fromAxisAngle(-ROT_SPEED, 0.0f, 0.0f, 1.0f);
            break;
        default:
            return;
    }
    orientation = delta * orientation;
    orientation.normalize();
    glutPostRedisplay();
}

// draw a unit cube centered at the origin, coloring each face differently
void drawCube() {
    glBegin(GL_QUADS);
    // +X face (right) — red
    glColor3f(1, 0, 0);
    glVertex3f( 0.5f, -0.5f, -0.5f);
    glVertex3f( 0.5f,  0.5f, -0.5f);
    glVertex3f( 0.5f,  0.5f,  0.5f);
    glVertex3f( 0.5f, -0.5f,  0.5f);

    // -X face (left) — green
    glColor3f(0, 1, 0);
    glVertex3f(-0.5f, -0.5f,  0.5f);
    glVertex3f(-0.5f,  0.5f,  0.5f);
    glVertex3f(-0.5f,  0.5f, -0.5f);
    glVertex3f(-0.5f, -0.5f, -0.5f);

    // +Y face (top) — blue
    glColor3f(0, 0, 1);
    glVertex3f(-0.5f,  0.5f, -0.5f);
    glVertex3f( 0.5f,  0.5f, -0.5f);
    glVertex3f( 0.5f,  0.5f,  0.5f);
    glVertex3f(-0.5f,  0.5f,  0.5f);

    // -Y face (bottom) — yellow
    glColor3f(1, 1, 0);
    glVertex3f(-0.5f, -0.5f,  0.5f);
    glVertex3f( 0.5f, -0.5f,  0.5f);
    glVertex3f( 0.5f, -0.5f, -0.5f);
    glVertex3f(-0.5f, -0.5f, -0.5f);

    // +Z face (front) — magenta
    glColor3f(1, 0, 1);
    glVertex3f(-0.5f, -0.5f,  0.5f);
    glVertex3f(-0.5f,  0.5f,  0.5f);
    glVertex3f( 0.5f,  0.5f,  0.5f);
    glVertex3f( 0.5f, -0.5f,  0.5f);

    // -Z face (back) — cyan
    glColor3f(0, 1, 1);
    glVertex3f( 0.5f, -0.5f, -0.5f);
    glVertex3f( 0.5f,  0.5f, -0.5f);
    glVertex3f(-0.5f,  0.5f, -0.5f);
    glVertex3f(-0.5f, -0.5f, -0.5f);
    glEnd();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // move it back so we can see it
    glTranslatef(0.0f, 0.0f, -5.0f);

    // build rotation matrix from quaternion
    float mat[16];
    orientation.toMatrix(mat);
    glMultMatrixf(mat);

    // draw
    drawCube();

    glutSwapBuffers();
}

void reshape(int w, int h) {
    if (h == 0) h = 1;
    float aspect = float(w) / float(h);

    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, aspect, 1.0, 100.0);

    glMatrixMode(GL_MODELVIEW);
}

void specialKeys(int key, int x, int y) {
    Quaternion<float> delta;

    switch (key) {
        case GLUT_KEY_UP:
            delta = Quaternion<float>::fromAxisAngle(-ROT_SPEED, 1.0f, 0.0f, 0.0f);
            break;
        case GLUT_KEY_DOWN:
            delta = Quaternion<float>::fromAxisAngle( ROT_SPEED, 1.0f, 0.0f, 0.0f);
            break;
        case GLUT_KEY_LEFT:
            delta = Quaternion<float>::fromAxisAngle(-ROT_SPEED, 0.0f, 1.0f, 0.0f);
            break;
        case GLUT_KEY_RIGHT:
            delta = Quaternion<float>::fromAxisAngle( ROT_SPEED, 0.0f, 1.0f, 0.0f);
            break;
        default:
            return;
    }

    orientation = delta * orientation;
    orientation.normalize();

    glutPostRedisplay();
}

int main(int argc, char** argv) {
    orientation = Quaternion<float>();

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Colored Quaternion-Rotated Cube");

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutSpecialFunc(specialKeys);
    glutKeyboardFunc(keyboard);

    glutMainLoop();
    return 0;
}
