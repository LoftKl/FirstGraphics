#ifndef QUATERNION_H
#define QUATERNION_H

#include <iostream>
#include <cmath>
#include <stdexcept> // Required for std::runtime_error
#include <limits>    // Required for std::numeric_limits (for floating point epsilon comparison)

/**
 * @class Quaternion
 * @brief A templated class for representing and manipulating quaternions.
 *
 * This class provides various operations essential for working with quaternions,
 * including arithmetic operations, normalization, conjugation, and conversions
 * to and from other rotation representations like axis-angle. It is templated
 * to support different numeric types such as float or double.
 *
 * @tparam T The numeric type for the quaternion's components (e.g., float, double).
 */
template<typename T>
class Quaternion {
private:
    T w, x, y, z;  // w is the scalar part, x, y, z are the vector parts

public:
    // Constructors
    
    /**
     * @brief Default constructor. Initializes a unit quaternion (1, 0, 0, 0).
     */
    Quaternion();
    
    /**
     * @brief Parameterized constructor.
     * @param w The scalar component.
     * @param x The x-component of the vector part.
     * @param y The y-component of the vector part.
     * @param z The z-component of the vector part.
     */
    Quaternion(T w, T x, T y, T z);
    
    /**
     * @brief Constructor from a scalar and a 3-element array.
     * @param scalar The scalar component (w).
     * @param vector A 3-element array containing the x, y, and z components.
     */
    Quaternion(T scalar, T vector[3]);
    
    /**
     * @brief Copy constructor.
     * @param other The quaternion to copy from.
     */
    Quaternion(const Quaternion<T>& other);

    // Destructor
    /**
     * @brief Destructor for the Quaternion class.
     */
    ~Quaternion();

    // Assignment operator
    /**
     * @brief Assignment operator.
     * @param other The quaternion to assign from.
     * @return A reference to the assigned quaternion.
     */
    Quaternion<T>& operator=(const Quaternion<T>& other);

    // Accessors
    /**
     * @brief Gets the scalar component (w).
     * @return The value of w.
     */
    T getW() const;
    
    /**
     * @brief Gets the x-component of the vector part.
     * @return The value of x.
     */
    T getX() const;
    
    /**
     * @brief Gets the y-component of the vector part.
     * @return The value of y.
     */
    T getY() const;
    
    /**
     * @brief Gets the z-component of the vector part.
     * @return The value of z.
     */
    T getZ() const;
    
    /**
     * @brief Gets the scalar component. Alias for getW().
     * @return The value of w.
     */
    T getScalar() const;
    
    /**
     * @brief Copies the vector part of the quaternion to a 3-element array.
     * @param vector A 3-element array to store the vector components.
     */
    void getVector(T vector[3]) const;

    // Mutators
    /**
     * @brief Sets the scalar component (w).
     * @param w The new value for w.
     */
    void setW(T w);
    
    /**
     * @brief Sets the x-component of the vector part.
     * @param x The new value for x.
     */
    void setX(T x);
    
    /**
     * @brief Sets the y-component of the vector part.
     * @param y The new value for y.
     */
    void setY(T y);
    
    /**
     * @brief Sets the z-component of the vector part.
     * @param z The new value for z.
     */
    void setZ(T z);
    
    /**
     * @brief Sets all components of the quaternion.
     * @param w The new scalar component.
     * @param x The new x-component.
     * @param y The new y-component.
     * @param z The new z-component.
     */
    void set(T w, T x, T y, T z);
    
    /**
     * @brief Sets the scalar component. Alias for setW().
     * @param scalar The new scalar component.
     */
    void setScalar(T scalar);
    
    /**
     * @brief Sets the vector part of the quaternion from a 3-element array.
     * @param vector A 3-element array containing the new x, y, and z components.
     */
    void setVector(T vector[3]);

    // Arithmetic operators
    /**
     * @brief Performs quaternion addition.
     * @param other The quaternion to add.
     * @return The result of the addition.
     */
    Quaternion<T> operator+(const Quaternion<T>& other) const;
    
    /**
     * @brief Performs quaternion subtraction.
     * @param other The quaternion to subtract.
     * @return The result of the subtraction.
     */
    Quaternion<T> operator-(const Quaternion<T>& other) const;
    
    /**
     * @brief Performs quaternion multiplication (Hamiltonian product).
     * @param other The quaternion to multiply by.
     * @return The result of the multiplication.
     */
    Quaternion<T> operator*(const Quaternion<T>& other) const;
    
    /**
     * @brief Performs scalar multiplication.
     * @param scalar The scalar to multiply by.
     * @return The result of the scalar multiplication.
     */
    Quaternion<T> operator*(T scalar) const;
    
    /**
     * @brief Performs scalar division.
     * @param scalar The scalar to divide by.
     * @return The result of the scalar division.
     * @throws std::runtime_error if division by zero occurs.
     */
    Quaternion<T> operator/(T scalar) const;

    // Compound assignment operators
    /**
     * @brief Adds another quaternion to this one.
     * @param other The quaternion to add.
     * @return A reference to this quaternion after addition.
     */
    Quaternion<T>& operator+=(const Quaternion<T>& other);
    
    /**
     * @brief Subtracts another quaternion from this one.
     * @param other The quaternion to subtract.
     * @return A reference to this quaternion after subtraction.
     */
    Quaternion<T>& operator-=(const Quaternion<T>& other);
    
    /**
     * @brief Multiplies this quaternion by another (Hamiltonian product).
     * @param other The quaternion to multiply by.
     * @return A reference to this quaternion after multiplication.
     */
    Quaternion<T>& operator*=(const Quaternion<T>& other);
    
    /**
     * @brief Multiplies this quaternion by a scalar.
     * @param scalar The scalar to multiply by.
     * @return A reference to this quaternion after scalar multiplication.
     */
    Quaternion<T>& operator*=(T scalar);
    
    /**
     * @brief Divides this quaternion by a scalar.
     * @param scalar The scalar to divide by.
     * @return A reference to this quaternion after scalar division.
     * @throws std::runtime_error if division by zero occurs.
     */
    Quaternion<T>& operator/=(T scalar);

    // Comparison operators
    /**
     * @brief Checks if two quaternions are approximately equal.
     * @param other The quaternion to compare with.
     * @return True if the quaternions are approximately equal, false otherwise.
     */
    bool operator==(const Quaternion<T>& other) const;
    
    /**
     * @brief Checks if two quaternions are not approximately equal.
     * @param other The quaternion to compare with.
     * @return True if the quaternions are not approximately equal, false otherwise.
     */
    bool operator!=(const Quaternion<T>& other) const;

    // Unary operators
    /**
     * @brief Performs unary negation on the quaternion (negates all components).
     * @return The negated quaternion.
     */
    Quaternion<T> operator-() const;

    // Quaternion operations
    /**
     * @brief Calculates the magnitude (or length) of the quaternion.
     * @return The magnitude of the quaternion.
     */
    T magnitude() const;
    
    /**
     * @brief Calculates the squared magnitude of the quaternion.
     * @return The squared magnitude. This is more efficient than magnitude() if you only
     * need to compare lengths, as it avoids a square root operation.
     */
    T magnitudeSquared() const;
    
    /**
     * @brief Returns a new normalized (unit) quaternion.
     * @return A new quaternion with a magnitude of 1.
     */
    Quaternion<T> normalized() const;
    
    /**
     * @brief Normalizes this quaternion in-place.
     */
    void normalize();
    
    /**
     * @brief Returns the conjugate of the quaternion.
     * @return A new quaternion that is the conjugate of this one.
     */
    Quaternion<T> conjugate() const;
    
    /**
     * @brief Calculates the inverse of the quaternion.
     * @return The inverse of the quaternion, defined as conjugate / magnitudeSquared.
     */
    Quaternion<T> inverse() const;
    
    /**
     * @brief Calculates the dot product of two quaternions.
     * @param other The other quaternion.
     * @return The scalar dot product.
     */
    T dot(const Quaternion<T>& other) const;

    // Rotation operations
    /**
     * @brief Creates a new quaternion from an axis-angle representation.
     * @param angle The angle of rotation in radians.
     * @param axisX The x-component of the rotation axis.
     * @param axisY The y-component of the rotation axis.
     * @param axisZ The z-component of the rotation axis.
     * @return A new quaternion representing the rotation.
     */
    static Quaternion<T> fromAxisAngle(T angle, T axisX, T axisY, T axisZ);
    
    /**
     * @brief Creates a new quaternion from Euler angles (roll, pitch, yaw).
     * @param roll The roll angle (rotation around the x-axis) in radians.
     * @param pitch The pitch angle (rotation around the y-axis) in radians.
     * @param yaw The yaw angle (rotation around the z-axis) in radians.
     * @return A new quaternion representing the rotation.
     */
    static Quaternion<T> fromEulerAngles(T roll, T pitch, T yaw);
    
    /**
     * @brief Converts the quaternion to its axis-angle representation.
     * @param angle A reference to store the rotation angle in radians.
     * @param axisX A reference to store the x-component of the rotation axis.
     * @param axisY A reference to store the y-component of the rotation axis.
     * @param axisZ A reference to store the z-component of the rotation axis.
     */
    void toAxisAngle(T& angle, T& axisX, T& axisY, T& axisZ) const;
    
    /**
     * @brief Converts the quaternion to Euler angles (roll, pitch, yaw).
     * @param roll A reference to store the roll angle (x-axis rotation) in radians.
     * @param pitch A reference to store the pitch angle (y-axis rotation) in radians.
     * @param yaw A reference to store the yaw angle (z-axis rotation) in radians.
     */
    void toEulerAngles(T& roll, T& pitch, T& yaw) const;

    /**
     * @brief Converts the quaternion to a 4x4 rotation matrix.
     * @param mat A pointer to a 16-element array to store the resulting 4x4 matrix.
     */
    void toMatrix(T mat[16]) const; // <-- Removed 'Quaternion<T>::' here
    
    // Interpolation
    /**
     * @brief Performs spherical linear interpolation between two quaternions.
     * @param q1 The starting quaternion.
     * @param q2 The ending quaternion.
     * @param t The interpolation factor, clamped between 0 and 1.
     * @return The interpolated quaternion.
     */
    static Quaternion<T> slerp(const Quaternion<T>& q1, const Quaternion<T>& q2, T t);
    
    /**
     * @brief Performs linear interpolation between two quaternions, followed by normalization.
     * @param q1 The starting quaternion.
     * @param q2 The ending quaternion.
     * @param t The interpolation factor, clamped between 0 and 1.
     * @return The interpolated and normalized quaternion.
     */
    static Quaternion<T> lerp(const Quaternion<T>& q1, const Quaternion<T>& q2, T t);

    // Utility functions
    /**
     * @brief Checks if the quaternion is a unit quaternion (magnitude is approximately 1).
     * @return True if the magnitude is close to 1, false otherwise.
     */
    bool isUnit() const;
    
    /**
     * @brief Checks if all quaternion components are approximately zero.
     * @return True if all components are close to zero, false otherwise.
     */
    bool isZero() const;
    
    /**
     * @brief Prints the quaternion components to the console.
     */
    void print() const;

    // Friend operators
    /**
     * @brief Left scalar multiplication.
     * @param scalar The scalar to multiply by.
     * @param q The quaternion to multiply.
     * @return The result of the multiplication.
     */
    template<typename U>
    friend Quaternion<U> operator*(U scalar, const Quaternion<U>& q);
    
    /**
     * @brief Stream insertion operator for printing a quaternion.
     * @param os The output stream.
     * @param q The quaternion to print.
     * @return The modified output stream.
     */
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Quaternion<U>& q);
    
    /**
     * @brief Stream extraction operator for reading a quaternion.
     * @param is The input stream.
     * @param q A reference to the quaternion to populate.
     * @return The modified input stream.
     */
    template<typename U>
    friend std::istream& operator>>(std::istream& is, Quaternion<U>& q);
};

// Template Implementation

// Constructors
template<typename T>
Quaternion<T>::Quaternion() : w(1), x(0), y(0), z(0){}

template<typename T>
Quaternion<T>::Quaternion(T w, T x, T y, T z) : w(w),x(x),y(y),z(z) {}

template<typename T>
Quaternion<T>::Quaternion(T scalar, T vector[3]): 
    w(scalar), x(vector[0]), y(vector[1]), z(vector[2]) {}

template<typename T>
Quaternion<T>::Quaternion(const Quaternion<T>& other): 
w(other.w), x(other.x), y(other.y), z(other.z) {} // Copy constructor

// Destructor
template<typename T>
Quaternion<T>::~Quaternion() {
    // TODO: Cleanup if needed (likely empty for basic types)
}

// Assignment operator
template<typename T>
Quaternion<T>& Quaternion<T>::operator=(const Quaternion<T>& other) {
    if (this != &other) { // Self-assignment check
        this->w = other.w;
        this->x = other.x;
        this->y = other.y;
        this->z = other.z;
    }
    return *this;
}

// Accessors
template<typename T>
T Quaternion<T>::getW() const {
    return w;
}

template<typename T>
T Quaternion<T>::getX() const {
    return x;
}

template<typename T>
T Quaternion<T>::getY() const {
    return y;
}

template<typename T>
T Quaternion<T>::getZ() const {
    return z;
}

template<typename T>
T Quaternion<T>::getScalar() const {
    return w;
}

template<typename T>
void Quaternion<T>::getVector(T vector[3]) const {
   vector[0] = x;
   vector[1] = y;
   vector[2] = z;
}

// Mutators
template<typename T>
void Quaternion<T>::setW(T w) {
    this->w = w;
}

template<typename T>
void Quaternion<T>::setX(T x) {
    this->x = x;
}

template<typename T>
void Quaternion<T>::setY(T y) {
    this->y = y;
}

template<typename T>
void Quaternion<T>::setZ(T z) {
    this->z = z;
}

template<typename T>
void Quaternion<T>::set(T w, T x, T y, T z) {
    this->w = w;
    this->x = x;
    this->y = y;
    this->z = z;
}

template<typename T>
void Quaternion<T>::setScalar(T scalar) {
    this->w = scalar;
}

template<typename T>
void Quaternion<T>::setVector(T vector[3]) {
    this->x = vector[0];
    this->y = vector[1];
    this->z = vector[2];
}

// Arithmetic operators
template<typename T>
Quaternion<T> Quaternion<T>::operator+(const Quaternion<T>& other) const {
    // Quaternion addition: (w1+w2, x1+x2, y1+y2, z1+z2)
    Quaternion<T> result(
        this->w + other.w, // Add w components
        this->x + other.x, // Add x components
        this->y + other.y, // Add y components
        this->z + other.z  // Add z components
    );
    return result;
}

template<typename T>
Quaternion<T> Quaternion<T>::operator-(const Quaternion<T>& other) const {
        Quaternion<T> result(
        this->w - other.w,
        this->x - other.x, 
        this->y - other.y,
        this->z - other.z  
    );
    return result;
}

template<typename T>
Quaternion<T> Quaternion<T>::operator*(const Quaternion<T>& other) const {
    T new_w = this->w * other.w - this->x * other.x - this->y * other.y - this->z * other.z;
    T new_x = this->w * other.x + this->x * other.w + this->y * other.z - this->z * other.y;
    T new_y = this->w * other.y - this->x * other.z + this->y * other.w + this->z * other.x;
    T new_z = this->w * other.z + this->x * other.y - this->y * other.x + this->z * other.w;

    return Quaternion<T>(new_w, new_x, new_y, new_z);
}

template<typename T>
Quaternion<T> Quaternion<T>::operator*(T scalar) const {
    // Multiply each component of the quaternion by the scalar
    return Quaternion<T>(
        this->w * scalar,
        this->x * scalar,
        this->y * scalar,
        this->z * scalar
    );
}



template<typename T>
Quaternion<T> Quaternion<T>::operator/(T scalar) const {
    // Check for division by zero
    // For floating-point types, compare to a small epsilon rather than exact zero
    // to account for floating-point inaccuracies.
    // For integer types, a direct comparison to zero is sufficient.
    if (scalar == static_cast<T>(0)) {
        // If T is a floating-point type, also check against a small epsilon
        // (std::numeric_limits<T>::epsilon() provides a good default for comparing floats)
        if (std::is_floating_point<T>::value && std::abs(scalar) < std::numeric_limits<T>::epsilon()) {
            throw std::runtime_error("Quaternion division by zero (scalar is too close to zero).");
        } else if (!std::is_floating_point<T>::value) { // For integer types
            throw std::runtime_error("Quaternion division by zero (scalar is exactly zero).");
        }
    }

    // If not zero, perform the division component-wise
    return Quaternion<T>(
        this->w / scalar,
        this->x / scalar,
        this->y / scalar,
        this->z / scalar
    );
}


// Compound assignment operators
template<typename T>
Quaternion<T>& Quaternion<T>::operator+=(const Quaternion<T>& other) {
    this->w += other.w;
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;

    return *this;
}

template<typename T>
Quaternion<T>& Quaternion<T>::operator-=(const Quaternion<T>& other) {
    this->w -= other.w;
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;

    return *this;
}

template<typename T>
Quaternion<T>& Quaternion<T>::operator*=(const Quaternion<T>& other) {
    T temp_w = this->w;
    T temp_x = this->x;
    T temp_y = this->y;
    T temp_z = this->z;

    // Apply the Hamiltonian product formula
    this->w = temp_w * other.w - temp_x * other.x - temp_y * other.y - temp_z * other.z;
    this->x = temp_w * other.x + temp_x * other.w + temp_y * other.z - temp_z * other.y;
    this->y = temp_w * other.y - temp_x * other.z + temp_y * other.w + temp_z * other.x;
    this->z = temp_w * other.z + temp_x * other.y - temp_y * other.x + temp_z * other.w;

    return *this; 
}

template<typename T>
Quaternion<T>& Quaternion<T>::operator*=(T scalar) {
    this->w = this->w * scalar;
    this->x = this->x * scalar;
    this->y = this->y * scalar;
    this->z = this->z * scalar;
    return *this;
}

template<typename T>
Quaternion<T>& Quaternion<T>::operator/=(T scalar) {
    // Check for division by zero
    // For floating-point types, compare to a small epsilon rather than exact zero
    if (scalar == static_cast<T>(0)) {
        if (std::is_floating_point<T>::value && std::abs(scalar) < std::numeric_limits<T>::epsilon()) { 
            throw std::runtime_error("Quaternion division assignment by zero (scalar is too close to zero)."); 
        } else if (!std::is_floating_point<T>::value) { 
            throw std::runtime_error("Quaternion division assignment by zero (scalar is exactly zero)."); 
        }
    }

    // If not zero, perform the division component-wise and assign back
    this->w /= scalar; 
    this->x /= scalar; 
    this->y /= scalar; 
    this->z /= scalar; 

    return *this; 
}
// Comparison operators
template<typename T>
bool Quaternion<T>::operator==(const Quaternion<T>& other) const {
    const T epsilon_limit = std::numeric_limits<T>::epsilon() * 10;
    Quaternion<T> diff = *this - other;  

    return diff.magnitude() < epsilon_limit;
}

template<typename T>
bool Quaternion<T>::operator!=(const Quaternion<T>& other) const {
    // The inequality operator is simply the logical negation of the equality operator.
    return !(*this == other);
}

// Unary operators
template<typename T>
Quaternion<T> Quaternion<T>::operator-() const {
    return Quaternion<T>(-w, -x, -y, -z);
}

// Quaternion operations
template<typename T>
T Quaternion<T>::magnitude() const {
    return std::sqrt(w*w + x*x + y*y + z*z);
}

template<typename T>
T Quaternion<T>::magnitudeSquared() const {
    return w*w + x*x + y*y + z*z;
}

template<typename T>
Quaternion<T> Quaternion<T>::normalized() const {
    T mag = magnitude();

    if (mag > std::numeric_limits<T>::epsilon()) {
        return *this / mag;
    }
    // If the magnitude is zero, return the zero quaternion to avoid division by zero.
    // Or, you could return the original quaternion, as it's already "normalized"
    // to a zero vector.
    return Quaternion<T>(0, 0, 0, 0); 
}

template<typename T>
void Quaternion<T>::normalize() {
    *this = normalized();
}

template<typename T>
Quaternion<T> Quaternion<T>::conjugate() const {
    return Quaternion<T>(w, -x, -y, -z);
}

template<typename T>
Quaternion<T> Quaternion<T>::inverse() const {
    // TODO: Return inverse (conjugate / magnitude squared)
    Quaternion<T> conj = conjugate();
    T mag_sqr = magnitudeSquared();
    return conj / mag_sqr;
}

template<typename T>
T Quaternion<T>::dot(const Quaternion<T>& other) const {
    return w * other.w + 
           x * other.x + 
           y * other.y + 
           z * other.z;
}

// Rotation operations
template<typename T>
Quaternion<T> Quaternion<T>::fromAxisAngle(T angle, T axisX, T axisY, T axisZ) {
    T half_angle = angle / 2.0;
    T sin_half_angle = std::sin(half_angle);
    T cos_half_angle = std::cos(half_angle);

    Quaternion<T> q(cos_half_angle,
                    axisX * sin_half_angle,
                    axisY * sin_half_angle,
                    axisZ * sin_half_angle);
    q.normalize();
    return q;
}

template<typename T>
Quaternion<T> Quaternion<T>::fromEulerAngles(T roll, T pitch, T yaw) {
    // The rotation order is Z-Y-X (yaw, then pitch, then roll).
    // The formula for a quaternion from Euler angles is derived by multiplying
    // three separate quaternions for each axis rotation.
    
    // Calculate half angles
    T halfRoll = roll * static_cast<T>(0.5);
    T halfPitch = pitch * static_cast<T>(0.5);
    T halfYaw = yaw * static_cast<T>(0.5);

    // Calculate sin and cos for each half angle
    T cosRoll = std::cos(halfRoll);
    T sinRoll = std::sin(halfRoll);
    T cosPitch = std::cos(halfPitch);
    T sinPitch = std::sin(halfPitch);
    T cosYaw = std::cos(halfYaw);
    T sinYaw = std::sin(halfYaw);

    // Apply the combined formula
    T new_w = cosYaw * cosPitch * cosRoll + sinYaw * sinPitch * sinRoll;
    T new_x = cosYaw * cosPitch * sinRoll - sinYaw * sinPitch * cosRoll;
    T new_y = cosYaw * sinPitch * cosRoll + sinYaw * cosPitch * sinRoll;
    T new_z = sinYaw * cosPitch * cosRoll - cosYaw * sinPitch * sinRoll;
    
    return Quaternion<T>(new_w, new_x, new_y, new_z);
}

template<typename T>
void Quaternion<T>::toAxisAngle(T& angle, T& axisX, T& axisY, T& axisZ) const {
    Quaternion<T> q = this->normalized(); // ensure the quaternion is unit length

    // Clamp w to [-1, 1] to avoid domain errors in acos due to numerical inaccuracies
    T clamped_w = std::max(std::min(q.w, static_cast<T>(1)), static_cast<T>(-1));
    angle = 2 * std::acos(clamped_w); // angle in radians

    T sin_half_angle = std::sqrt(1 - clamped_w * clamped_w);

    // Avoid division by zero; if sin is too small, the axis is arbitrary (default to x-axis)
    if (sin_half_angle < std::numeric_limits<T>::epsilon()) {
        axisX = 1;
        axisY = 0;
        axisZ = 0;
    } else {
        axisX = q.x / sin_half_angle;
        axisY = q.y / sin_half_angle;
        axisZ = q.z / sin_half_angle;
    }
}


// Rotation operations
template<typename T>
void Quaternion<T>::toEulerAngles(T& roll, T& pitch, T& yaw) const {
    T q_w = this->w;
    T q_x = this->x;
    T q_y = this->y;
    T q_z = this->z;

    // Use a normalized quaternion to ensure the input is valid.
    T q_w_norm = q_w;
    T q_x_norm = q_x;
    T q_y_norm = q_y;
    T q_z_norm = q_z;
    T magnitude = std::sqrt(q_w_norm*q_w_norm + q_x_norm*q_x_norm + q_y_norm*q_y_norm + q_z_norm*q_z_norm);
    if (magnitude > std::numeric_limits<T>::epsilon()) {
        q_w_norm /= magnitude;
        q_x_norm /= magnitude;
        q_y_norm /= magnitude;
        q_z_norm /= magnitude;
    }

    // Roll (x-axis rotation)
    T sinr_cosp = 2.0 * (q_w_norm * q_x_norm + q_y_norm * q_z_norm);
    T cosr_cosp = 1.0 - 2.0 * (q_x_norm * q_x_norm + q_y_norm * q_y_norm);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // Pitch (y-axis rotation)
    T sinp = 2.0 * (q_w_norm * q_y_norm - q_z_norm * q_x_norm);
    if (std::abs(sinp) >= 1)
        // Use asin with clamp to handle potential floating point errors and gimbal lock.
        pitch = std::copysign(M_PI / 2.0, sinp); // 90 degrees or -90 degrees
    else
        pitch = std::asin(sinp);

    // Yaw (z-axis rotation)
    T siny_cosp = 2.0 * (q_w_norm * q_z_norm + q_x_norm * q_y_norm);
    T cosy_cosp = 1.0 - 2.0 * (q_y_norm * q_y_norm + q_z_norm * q_z_norm);
    yaw = std::atan2(siny_cosp, cosy_cosp);
}

template<typename T>
void Quaternion<T>::toMatrix(T mat[16]) const {
    T xx = x * x;
    T yy = y * y;
    T zz = z * z;
    T xy = x * y;
    T xz = x * z;
    T yz = y * z;
    T wx = w * x;
    T wy = w * y;
    T wz = w * z;

    mat[0] = 1 - 2 * (yy + zz);
    mat[1] = 2 * (xy + wz);
    mat[2] = 2 * (xz - wy);
    mat[3] = 0;

    mat[4] = 2 * (xy - wz);
    mat[5] = 1 - 2 * (xx + zz);
    mat[6] = 2 * (yz + wx);
    mat[7] = 0;

    mat[8] = 2 * (xz + wy);
    mat[9] = 2 * (yz - wx);
    mat[10] = 1 - 2 * (xx + yy);
    mat[11] = 0;

    mat[12] = 0;
    mat[13] = 0;
    mat[14] = 0;
    mat[15] = 1;
}


template<typename T>
Quaternion<T> Quaternion<T>::slerp(const Quaternion<T>& q1, const Quaternion<T>& q2, T t) {
    // Clamp t to the range [0, 1]
    t = std::max(static_cast<T>(0), std::min(static_cast<T>(1), t));

    Quaternion<T> q2_prime = q2;
    T dot = q1.dot(q2);

    // If the dot product is negative, the quaternions are more than 90 degrees apart.
    // To take the shorter path, we negate one of the quaternions.
    if (dot < 0.0) {
        q2_prime = -q2;
        dot = -dot;
    }

    // If the quaternions are very close, we can use linear interpolation
    // to avoid potential division by zero.
    if (dot > static_cast<T>(1.0) - std::numeric_limits<T>::epsilon()) {
        return lerp(q1, q2_prime, t);
    }
    
    // Standard SLERP formula
    T theta = std::acos(dot);
    T sin_theta = std::sin(theta);

    // Handle potential division by zero
    if (sin_theta == 0) {
        return lerp(q1, q2_prime, t);
    }
    
    T c1 = std::sin((1 - t) * theta) / sin_theta;
    T c2 = std::sin(t * theta) / sin_theta;

    return (q1 * c1) + (q2_prime * c2);
}

template<typename T>
Quaternion<T> Quaternion<T>::lerp(const Quaternion<T>& q1, const Quaternion<T>& q2, T t) {
    // Clamp t to the range [0, 1]
    t = std::max(static_cast<T>(0), std::min(static_cast<T>(1), t));
    
    Quaternion<T> result = q1 + ((q2 - q1) * t);
    return result.normalized();
}

// Utility functions
template<typename T>
bool Quaternion<T>::isUnit() const {
    const T epsilon_limit = std::numeric_limits<T>::epsilon() * 10;
    return std::abs(magnitudeSquared() - static_cast<T>(1.0)) < epsilon_limit;
}

template<typename T>
bool Quaternion<T>::isZero() const {
    const T epsilon_limit = std::numeric_limits<T>::epsilon() * 10;
    return magnitudeSquared() < epsilon_limit;
}

template<typename T>
void Quaternion<T>::print() const {
    std::cout << "Quaternion(" << this->w << ", " << this->x << ", " << this->y << ", " << this->z << ")" << std::endl;
}

// Friend operators
template<typename T>
Quaternion<T> operator*(T scalar, const Quaternion<T>& q) {
    return q * scalar;
}

template<typename T>
std::ostream& operator<<(std::ostream& os, const Quaternion<T>& q) {
    os << "(" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ")";
    return os;
}

template<typename T>
std::istream& operator>>(std::istream& is, Quaternion<T>& q) {
    // Read components, possibly ignoring delimiters like commas and parentheses.
    is >> q.w >> q.x >> q.y >> q.z;
    return is;
}

#endif // QUATERNION_H
