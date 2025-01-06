#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <stddef.h>  // aligned_alloc için gerekli

// Hata kontrolü için makro
#define CHECK_NULL(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Error: Null pointer at %s:%d\n", __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

// Error checking macros
#define CHECK_MALLOC(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Memory allocation failed at %s:%d\n", __FILE__, __LINE__); \
        return false; \
    } \
} while(0)

// System Constants
#define AZIMUTH_MIN 0
#define AZIMUTH_MAX 360
#define ELEVATION_MIN 0
#define ELEVATION_MAX 90
#define AZIMUTH_RESOLUTION 1   // Degrees per step
#define ELEVATION_RESOLUTION 1 // Degrees per step
#define AZIMUTH_POINTS ((AZIMUTH_MAX - AZIMUTH_MIN) / AZIMUTH_RESOLUTION)
#define ELEVATION_POINTS ((ELEVATION_MAX - ELEVATION_MIN) / ELEVATION_RESOLUTION)
#define AZIMUTH_SEARCH_AREA 60
#define ELEVATION_SEARCH_AREA 30

// PSO Parameters
#define NUM_PARTICLES 4
#define MAX_ITERATIONS 10
#define INERTIA_MAX 0.9
#define INERTIA_MIN 0.4
#define C1 2.0
#define C2 2.0

// Signal Parameters
#define RSSI_MAX -40              // Maximum RSSI in dBm (very close range)
#define RSSI_MIN -85              // Minimum RSSI in dBm (very weak signal)
#define SNR_MAX 40                // Maximum SNR in dB (excellent signal)
#define SNR_MIN 10                // Minimum SNR in dB (unusable signal)
#define SIGNAL_QUALITY_THRESHOLD 0.4
#define NOISE_FLOOR -100
#define PATH_LOSS_EXPONENT 2.5  // Path loss exponent (2.0-4.0 typical range)
#define REFERENCE_DISTANCE 1.0  // Reference distance in meters
#define REFERENCE_POWER -30.0   // Reference power at reference distance

// Antenna Movement Parameters
#define MAX_ANGULAR_SPEED 30.0  // Degrees per second
#define SERVO_UPDATE_INTERVAL 0.1 // Seconds

// Kalman Filter Parameters
#define KALMAN_Q 0.01  // Process noise covariance
#define KALMAN_R 0.1   // Measurement noise covariance

// Environmental Factors
typedef struct {
    float temperature;     // Celsius
    float humidity;        // Percentage
    float interference;    // 0-1 scale
    float rain_intensity;  // mm/h
    float wind_speed;      // m/s
    float time_variation;  // Time-based variation
    float atmospheric_loss;    // Atmospheric absorption loss (dB/km)
    float multipath_factor;    // Multipath fading factor
    float doppler_shift;       // Doppler shift due to UAV motion
    float terrain_factor;      // Terrain/obstacle effects
    float atmosphericPressure;    // hPa
    float airDensity;            // kg/m³
    float visibilityRange;       // metre
    float cloudCover;            // 0-1 arası
    float turbulenceIntensity;   // 0-1 arası
    float solarRadiation;        // W/m²
    float magneticInterference;  // Tesla
} EnvironmentalConditions;

// PSO Particle Structure
typedef struct {
    float position[2];        // Current position [azimuth, elevation]
    float velocity[2];        // Current velocity
    float bestPosition[2];    // Personal best position
    float fitness;            // Current fitness
    float bestFitness;        // Personal best fitness
    int stagnationCount;      // Iterations without improvement
} Particle;

// System State Structure
typedef struct {
    float globalBestPosition[2];
    float globalBestFitness;
    float previousBestFitness;
    int searchMode;           // 0: Wide search, 1: Narrow search
    int totalScans;
    time_t startTime;
    float servoAngles[2];     // Current servo angles
    float targetAngles[2];    // Target servo angles
    float kalmanState[2];     // Kalman filter state for [azimuth, elevation]
    float kalmanCovariance[2];// Kalman filter covariance for [azimuth, elevation]
    float timeSinceLastUpdate;
} SystemState;

// UAV Structure
typedef struct {
    float position[3];        // Current position [x, y, z]
    float velocity[3];        // Velocity components [vx, vy, vz]
    float pathTime;           // Time along the path
    float acceleration[3];     // Acceleration components [ax, ay, az]
    float heading;            // UAV heading angle
    float pitch;             // UAV pitch angle
    float roll;              // UAV roll angle
    float turnRate;          // Turn rate (rad/s)
    enum {
        TAKEOFF,
        CRUISE,
        HOVER,
        LANDING,
        LANDED
    } flightState;
    float throttle;           // 0-100%
    float enginePower[4];     // Her motor için güç
    float batteryLevel;       // 0-100%
    float accelerometer[3];   // x,y,z eksenleri
    float gyroscope[3];       // roll, pitch, yaw rates
    float altitude;           // Deniz seviyesinden yükseklik
    float airSpeed;           // Hava hızı
    float groundSpeed;        // Yer hızı
    float windEffect[3];      // Rüzgar etkisi vektörü
    float targetAltitude;
    float targetHeading;
    float flightPath[3];      // Hedef yol noktaları
    int currentWaypoint;
    float signalStrength;     // dBm cinsinden
    float signalQuality;      // 0-1 arası
} UAV;

// 3D Matrix yapısı
typedef struct {
    int dimensions[3];           // x,y,z boyutları
    float cellSize;             // metre cinsinden hücre boyutu
    float*** signalStrength;    // Her hücre için sinyal gücü
    float*** noiseLevel;        // Her hücre için gürültü seviyesi
} FlightMatrix;

// Kalman Filtresi genişletilmiş yapısı
typedef struct {
    float state[6];             // [x, y, z, vx, vy, vz]
    float covariance[6][6];     // Kovaryans matrisi
    float processNoise[6][6];   // Q matrisi
    float measurementNoise[3][3]; // R matrisi
} KalmanFilter;

// Cache-friendly veri yapısı
typedef struct {
    float positions[NUM_PARTICLES][2];  // Structure of Arrays (SoA) yapısı
    float velocities[NUM_PARTICLES][2];
    float fitness[NUM_PARTICLES];
    float bestPositions[NUM_PARTICLES][2];
    float bestFitness[NUM_PARTICLES];
} ParticleSystem;

// Optimize edilmiş sinyal haritası yapısı
typedef struct {
    float* rssiData;  // Tek boyutlu dizi
    float* snrData;   // Tek boyutlu dizi
    float** rssiMap;  // Pointer array
    float** snrMap;   // Pointer array
} SignalMaps;

// Trigonometrik tablo yapısı
typedef struct {
    float sinTable[360];
    float cosTable[360];
    bool initialized;
} TrigTables;

// Thread verileri için yapı
typedef struct {
    int startIdx;
    int endIdx;
    ParticleSystem* particles;
    const SystemState* state;
    const FlightMatrix* matrix;
} ThreadData;

// Global değişkenler
static TrigTables trigTables = {0};
static SignalMaps signalMaps = {0};

// Function Prototypes
void initializeSystem(SystemState* state);
void initializeEnvironment(EnvironmentalConditions* env);
void initializeParticles(Particle* particles, float x_min, float x_max, float y_min, float y_max);
void updateEnvironmentalConditions(EnvironmentalConditions* env);
void updateUAVPosition(UAV* uav, const EnvironmentalConditions* env, float deltaTime);
float calculateRSSI(float azimuth, float elevation, const UAV* uav, const EnvironmentalConditions* env);
float calculateSNR(float azimuth, float elevation, const UAV* uav, const EnvironmentalConditions* env);
float calculateFitness(float rssi, float snr);
float randomFloat(float min, float max);
float degreeToRadian(float degree);
float radianToDegree(float radian);
void runPSO(SystemState* state, EnvironmentalConditions* env, UAV* uav, int mode, float** rssiMap, float** snrMap);
void printSimulationStatus(const SystemState* state, const EnvironmentalConditions* env, const UAV* uav);
void calculateAzimuthElevationToUAV(const UAV* uav, float* azimuth, float* elevation);
void updateServoAngles(SystemState* state, float deltaTime);
float wrapAngle360(float angle);
void applyKalmanFilter(SystemState* state);
void updatePSOParameters(float* w, int iteration, int max_iterations, Particle* particles);
float calculateAntennaGain(float azimuthDiff, float elevationDiff);
void updateSignalMap(float** rssiMap, float** snrMap, const UAV* uav, const EnvironmentalConditions* env);
bool allocateSignalMaps(float*** rssiMap, float*** snrMap);
void freeSignalMaps(float** rssiMap, float** snrMap);
void simulateAntennaTracking();
float calculateDetailedAntennaGain(float azimuthError, float elevationError);
float calculateWaterVaporAttenuation(float humidity, float temperature);
float calculateRicianFading(float K, float multipathFactor);
float calculateUrbanLoss(float terrainFactor, float distance);
float calculateDopplerShift(const UAV* uav, float lambda);
float calculateInterference(float interferenceLevel);
float calculateSystemNoise(void);
void initializeFlightMatrix(FlightMatrix* matrix, int x, int y, int z, float cellSize);
void updateUAVDynamics(UAV* uav, const EnvironmentalConditions* env, float deltaTime);
void performTakeoff(UAV* uav, float deltaTime);
void updateFlightPath(UAV* uav, const FlightMatrix* matrix);
void applyExtendedKalmanFilter(KalmanFilter* kf, UAV* uav, float* measurements);
float calculateDetailedRSSI(const UAV* uav, const EnvironmentalConditions* env, const FlightMatrix* matrix);
float calculateDetailedSNR(const UAV* uav, const EnvironmentalConditions* env, const FlightMatrix* matrix);
void updateSignalPropagation(FlightMatrix* matrix, const UAV* uav, const EnvironmentalConditions* env);

// Eksik fonksiyon prototipleri
void performTakeoff(UAV* uav, float deltaTime);
void performCruise(UAV* uav, float deltaTime, const EnvironmentalConditions* env);
void applyAtmosphericEffects(UAV* uav, const EnvironmentalConditions* env);
void updateEngineForces(UAV* uav);
void updatePhysics(UAV* uav, float deltaTime);
float calculateDistance(const UAV* uav);
float calculatePathLoss(float distance, const EnvironmentalConditions* env);
float calculateAtmosphericLoss(const EnvironmentalConditions* env, float distance);
float calculateDirectionalLoss(const UAV* uav);
float calculateOtherLosses(const EnvironmentalConditions* env);
void predictState(KalmanFilter* kf, const UAV* uav);
void updateMeasurement(KalmanFilter* kf, const float* measurements);
void updateState(KalmanFilter* kf, UAV* uav);
int calculateOptimalParticleCount(const SystemState* state);
void optimizeSearchSpace(SystemState* state, const UAV* uav);
void updateParticles(SystemState* state, const FlightMatrix* matrix);
void updateGlobalBest(SystemState* state);
bool checkConvergence(const SystemState* state);
void initializeCircularPath(UAV* uav);

// Eksik fonksiyon prototipleri ekle
void applyWindEffects(UAV* uav, const EnvironmentalConditions* env, float deltaTime);
void updateAttitudeAngles(UAV* uav);
float calculateEnvironmentalLoss(const EnvironmentalConditions* env, float distance);
void predictKalmanState(KalmanFilter* kf, float F[6][6]);
void updateKalmanMeasurement(KalmanFilter* kf, const float* measurements);
void applyKalmanToUAV(KalmanFilter* kf, UAV* uav);
float wrapAngle180(float angle);
float calculateAntennaPatternLoss(float azimuthError, float elevationError);

// UAV uçuş parametreleri
#define MAX_PITCH_ANGLE 30.0f        // Maksimum yunuslama açısı (derece)
#define MAX_ROLL_ANGLE 45.0f         // Maksimum yatış açısı (derece)
#define MAX_YAW_RATE 45.0f           // Maksimum dönüş hızı (derece/saniye)
#define MAX_VERTICAL_SPEED 5.0f      // Maksimum dikey hız (m/s)
#define ACCELERATION_SMOOTHING 0.15f  // Hızlanma yumuşatma faktörü
#define ATTITUDE_SMOOTHING 0.1f      // Açısal hareket yumuşatma faktörü
#define CRUISE_ALTITUDE 100.0f       // Seyir irtifası (m)
#define MIN_TURN_RADIUS 30.0f        // Minimum dönüş yarıçapı (m)

// UAV Flight Parameters
#define UAV_MIN_SPEED 20.0f         // Minimum hız (m/s)
#define UAV_MAX_SPEED 30.0f         // Maksimum hız (m/s)
#define UAV_MIN_ALTITUDE 100.0f     // Minimum irtifa (m)
#define UAV_MAX_ALTITUDE 300.0f     // Maksimum irtifa (m)
#define UAV_CLIMB_RATE 5.0f         // Tırmanma hızı (m/s)
#define UAV_TURN_RATE 0.2f          // Dönüş hızı (rad/s)
#define CIRCLE_MIN_RADIUS 100.0f    // Minimum dönüş yarıçapı (m)
#define CIRCLE_MAX_RADIUS 300.0f    // Maksimum dönüş yarıçapı (m)

// Uçuş durumu kontrolü için yeni fonksiyonlar
void smoothAttitudeTransition(UAV* uav, float targetPitch, float targetRoll, float deltaTime);
void maintainFlightEnvelope(UAV* uav);
void updateFlightDynamics(UAV* uav, const EnvironmentalConditions* env, float deltaTime);
void calculateTargetWaypoint(UAV* uav, float* targetPos);

// UAV dinamiklerini güncelleme
void updateUAVDynamics(UAV* uav, const EnvironmentalConditions* env, float deltaTime) {
    // Uçuş durumuna göre davranış
    switch(uav->flightState) {
        case TAKEOFF:
            performTakeoff(uav, deltaTime);
            break;
        case CRUISE:
            performCruise(uav, deltaTime, env);
            break;
        // ...diğer durumlar...
    }
    
    // Atmosferik etkileri uygula
    applyAtmosphericEffects(uav, env);
    
    // Motor dinamiklerini güncelle
    updateEngineForces(uav);
    
    // Fizik motoru güncellemesi
    updatePhysics(uav, deltaTime);
}

// Detaylı RSSI hesaplama
float calculateDetailedRSSI(const UAV* uav, const EnvironmentalConditions* env, const FlightMatrix* matrix) {
    float baseRSSI = -30.0f; // Maksimum sinyal gücü
    
    // Mesafe kaybı
    float distance = calculateDistance(uav);
    float pathLoss = calculatePathLoss(distance, env);
    
    // Atmosferik kayıplar
    float atmosphericLoss = calculateAtmosphericLoss(env, distance);
    
    // Yönsel kayıplar
    float directionalLoss = calculateDirectionalLoss(uav);
    
    // Diğer kayıplar
    float otherLosses = calculateOtherLosses(env);
    
    return baseRSSI - pathLoss - atmosphericLoss - directionalLoss - otherLosses;
}

// PSO algoritması optimizasyonu
void optimizedPSO(SystemState* state, const FlightMatrix* matrix, const UAV* uav) {
    // Parçacık sayısını dinamik olarak ayarla
    int numParticles = calculateOptimalParticleCount(state);
    
    // Arama alanını optimize et
    optimizeSearchSpace(state, uav);
    
    // PSO iterasyonları
    for(int i = 0; i < MAX_ITERATIONS; i++) {
        // Parçacıkları güncelle
        updateParticles(state, matrix);
        
        // Global en iyiyi güncelle
        updateGlobalBest(state);
        
        // Yakınsama kontrolü
        if(checkConvergence(state)) break;
    }
}

int main() {
    simulateAntennaTracking();
    return 0;
}

// Yeni parametre tanımları
#define SIMULATION_STEPS 10000     // Maksimum simülasyon adımı
#define INITIAL_ALTITUDE 0.0f      // Başlangıç irtifası (m)
#define SPIRAL_CLIMB_RATE 2.0f    // Spiral tırmanış hızı (m/s)
#define MAX_ALTITUDE 500.0f        // Maksimum tırmanma irtifası (m)
#define CIRCULAR_RADIUS 200.0f     // Dairesel uçuş yarıçapı (m)

// RSSI ve SNR parametrelerini gerçekçi değerlere ayarla
#define RSSI_MAX -40              // Maksimum RSSI (dBm)
#define RSSI_MIN -85             // Minimum RSSI (dBm)
#define SNR_MAX 40               // Maksimum SNR (dB)
#define SNR_MIN 10               // Minimum SNR (dB)
#define SIGNAL_QUALITY_THRESHOLD 0.4  // Kalite eşiği

// Simulate the antenna tracking system
void simulateAntennaTracking() {
    srand(time(NULL) ^ (getpid() << 16));
    SystemState state = {0};
    EnvironmentalConditions env = {0};
    UAV uav = {0};

    // Initialize values
    initializeSystem(&state);
    initializeEnvironment(&env);

    // Initialize UAV position and velocity
    uav.position[0] = 1000;  // x (meters)
    uav.position[1] = 0;     // y (meters)
    uav.position[2] = 100;   // z (altitude in meters)
    uav.velocity[0] = 0;     // vx (m/s)
    uav.velocity[1] = 50;    // vy (m/s)
    uav.velocity[2] = 0;     // vz (m/s)
    uav.pathTime = 0;

    // Allocate signal maps
    float** rssiMap;
    float** snrMap;
    allocateSignalMaps(&rssiMap, &snrMap);

    float deltaTime = 0.5; // Time step in seconds

    // Başlangıç durumu ayarla
    uav.position[0] = 0;
    uav.position[1] = 0;
    uav.position[2] = INITIAL_ALTITUDE;
    uav.flightState = TAKEOFF;
    uav.batteryLevel = 100.0f;
    bool takeoffCompleted = false;
    static float currentAltitude = INITIAL_ALTITUDE;
    float pathAngle = 0.0f;
    int step = 0;

    while (step++ < SIMULATION_STEPS) {
        // UAV durumunu kontrol et ve güncelle
        if (uav.flightState == TAKEOFF && uav.position[2] >= CRUISE_ALTITUDE) {
            takeoffCompleted = true;
            uav.flightState = CRUISE;
            initializeCircularPath(&uav);
        }

        // Spiral yükseliş hareketi
        if (uav.flightState == CRUISE) {
            float omega = 2.0f * M_PI / 60.0f; // 1 dakikada bir tur
            pathAngle += omega * deltaTime;
            
            // Spiral hareket
            uav.position[0] = CIRCULAR_RADIUS * cos(pathAngle);
            uav.position[1] = CIRCULAR_RADIUS * sin(pathAngle);
            currentAltitude = fmin(currentAltitude + SPIRAL_CLIMB_RATE * deltaTime, MAX_ALTITUDE);
            uav.position[2] = currentAltitude;
            
            // Hız vektörünü güncelle
            uav.velocity[0] = -CIRCULAR_RADIUS * omega * sin(pathAngle);
            uav.velocity[1] = CIRCULAR_RADIUS * omega * cos(pathAngle);
            uav.velocity[2] = (currentAltitude < MAX_ALTITUDE) ? SPIRAL_CLIMB_RATE : 0;
        }

        // Update environmental conditions
        updateEnvironmentalConditions(&env);

        // Update UAV position
        updateUAVPosition(&uav, &env, deltaTime);

        // Update signal maps
        updateSignalMap(rssiMap, snrMap, &uav, &env);

        printf("\nStarting PSO to track UAV...\n");
        runPSO(&state, &env, &uav, state.searchMode, rssiMap, snrMap);

        // Apply Kalman filter to target angles
        applyKalmanFilter(&state);

        // Update servo angles towards target angles with movement constraints
        updateServoAngles(&state, deltaTime);

        // Print simulation status
        printSimulationStatus(&state, &env, &uav);

        // Signal quality threshold check
        if (state.globalBestFitness < SIGNAL_QUALITY_THRESHOLD) {
            printf("\nSignal quality degraded! Restarting wide area search...\n");
            state.globalBestFitness = -INFINITY;
            state.searchMode = 0; // Switch to wide search
        } else {
            state.previousBestFitness = state.globalBestFitness;
            state.searchMode = 1; // Continue narrow search
        }

        usleep((int)(deltaTime * 1e6)); // Wait for deltaTime seconds
    }

    // Free signal maps
    freeSignalMaps(rssiMap, snrMap);
}

// Initialize system state
void initializeSystem(SystemState* state) {
    state->globalBestFitness = -INFINITY;
    state->searchMode = 0;
    state->totalScans = 0;
    state->startTime = time(NULL);
    state->globalBestPosition[0] = 0;
    state->globalBestPosition[1] = 0;
    state->servoAngles[0] = 0;
    state->servoAngles[1] = 0;
    state->targetAngles[0] = 0;
    state->targetAngles[1] = 0;
    state->kalmanState[0] = 0;
    state->kalmanState[1] = 0;
    state->kalmanCovariance[0] = 1;
    state->kalmanCovariance[1] = 1;
    state->timeSinceLastUpdate = 0.0;
}

// Initialize environmental conditions
void initializeEnvironment(EnvironmentalConditions* env) {
    env->temperature = 25.0;    // 25°C
    env->humidity = 60.0;       // 60%
    env->interference = 0.1;    // Low interference
    env->rain_intensity = 0.0;  // No rain
    env->wind_speed = 5.0;      // Moderate wind
    env->time_variation = 0.0;  // Start at time 0
    env->atmospheric_loss = 0.1; // dB/km
    env->multipath_factor = 0.5; // Multipath fading factor
    env->doppler_shift = 0.1;    // Doppler shift
    env->terrain_factor = 0.2;   // Terrain/obstacle effects
    env->atmosphericPressure = 1013.25;    // hPa
    env->airDensity = 1.225;            // kg/m³
    env->visibilityRange = 10000;       // metre
    env->cloudCover = 0.5;            // 0-1 arası
    env->turbulenceIntensity = 0.1;   // 0-1 arası
    env->solarRadiation = 500;        // W/m²
    env->magneticInterference = 0.00005;  // Tesla
}

// Update environmental conditions
void updateEnvironmentalConditions(EnvironmentalConditions* env) {
    // Simulate environmental changes over time
    env->temperature += randomFloat(-0.2, 0.2);
    env->humidity += randomFloat(-0.3, 0.3);
    env->interference = 0.1 + 0.2 * sin(env->time_variation);
    env->rain_intensity = fmax(0.0, 3.0 * sin(env->time_variation / 5));
    env->wind_speed += randomFloat(-2, 2);

    // Ensure values stay within realistic bounds
    env->temperature = fmax(-10, fmin(40, env->temperature));
    env->humidity = fmax(0, fmin(100, env->humidity));
    env->interference = fmax(0, fmin(1, env->interference));
    env->rain_intensity = fmax(0, fmin(20, env->rain_intensity));
    env->wind_speed = fmax(0, fmin(20, env->wind_speed));

    env->time_variation += 0.1;
}


// Yumuşak açısal geçiş
void smoothAttitudeTransition(UAV* uav, float targetPitch, float targetRoll, float deltaTime) {
    float pitchDiff = targetPitch - uav->pitch;
    float rollDiff = targetRoll - uav->roll;

    // Maksimum açısal hız sınırlaması
    float maxRotation = degreeToRadian(30.0f) * deltaTime;

    uav->pitch += fmax(-maxRotation, fmin(maxRotation, pitchDiff * ATTITUDE_SMOOTHING));
    uav->roll += fmax(-maxRotation, fmin(maxRotation, rollDiff * ATTITUDE_SMOOTHING));
}

// Uçuş zarfı sınırlarını koruma
void maintainFlightEnvelope(UAV* uav) {
    // Yunuslama açısı sınırlaması
    float maxPitch = degreeToRadian(MAX_PITCH_ANGLE);
    uav->pitch = fmax(-maxPitch, fmin(maxPitch, uav->pitch));

    // Yatış açısı sınırlaması
    float maxRoll = degreeToRadian(MAX_ROLL_ANGLE);
    uav->roll = fmax(-maxRoll, fmin(maxRoll, uav->roll));

    // Minimum irtifa kontrolü
    float minAltitude = 20.0f;
    if(uav->position[2] < minAltitude) {
        uav->position[2] = minAltitude;
        uav->velocity[2] = fmax(0.0f, uav->velocity[2]);
    }
}

// Convert degrees to radians
float degreeToRadian(float degree) {
    return degree * M_PI / 180.0;
}

// Convert radians to degrees
float radianToDegree(float radian) {
    return radian * 180.0 / M_PI;
}

// Calculate azimuth and elevation angles from antenna to UAV
void calculateAzimuthElevationToUAV(const UAV* uav, float* azimuth, float* elevation) {
    // Antenna is at origin (0,0,0)
    float dx = uav->position[0];
    float dy = uav->position[1];
    float dz = uav->position[2];

    float distanceXY = sqrt(dx * dx + dy * dy);
    float distance = sqrt(distanceXY * distanceXY + dz * dz);

    *elevation = asin(dz / distance) * (180.0 / M_PI);
    *azimuth = atan2(dy, dx) * (180.0 / M_PI);
    if (*azimuth < 0) *azimuth += 360.0;
}

// Calculate antenna gain based on angle differences
float calculateAntennaGain(float azimuthDiff, float elevationDiff) {
    // Simplified antenna gain pattern with main lobe and side lobes
    float mainLobeWidthAzimuth = 10.0; // degrees
    float mainLobeWidthElevation = 10.0; // degrees
    float sideLobeLevel = -20.0; // dB

    // Calculate azimuth gain
    float azimuthGain;
    if (azimuthDiff <= mainLobeWidthAzimuth / 2.0) {
        azimuthGain = 0.0; // 0 dB in main lobe
    } else {
        azimuthGain = sideLobeLevel; // Side lobe level
    }

    // Calculate elevation gain
    float elevationGain;
    if (elevationDiff <= mainLobeWidthElevation / 2.0) {
        elevationGain = 0.0; // 0 dB in main lobe
    } else {
        elevationGain = sideLobeLevel; // Side lobe level
    }

    // Total antenna gain
    float totalGain = azimuthGain + elevationGain;

    // Convert gain from dB to linear scale
    return pow(10.0, totalGain / 10.0);
}

// Calculate RSSI based on antenna orientation and UAV position
float calculateRSSI(float azimuth, float elevation, const UAV* uav, const EnvironmentalConditions* env) {
    float distance = calculateDistance(uav);
    
    // Friis denklemi ile temel yol kaybı
    float wavelength = 0.125; // 2.4 GHz için
    float pathLoss = 20 * log10(4 * M_PI * distance / wavelength);
    
    // Anten kazancı
    float antennaGain = 15.0; // dBi
    float targetAzimuth, targetElevation;
    calculateAzimuthElevationToUAV(uav, &targetAzimuth, &targetElevation);
    float azimuthError = fabs(wrapAngle180(azimuth - targetAzimuth));
    float elevationError = fabs(elevation - targetElevation);
    
    // Yönelim kaybı
    float pointingLoss = calculateAntennaPatternLoss(azimuthError, elevationError);
    
    // Toplam RSSI hesaplama
    float txPower = 20.0; // dBm
    float rssi = txPower + antennaGain - pathLoss - pointingLoss;
    
    // Mesafeye bağlı gürültü ekle
    float noise = randomFloat(-2.0, 2.0) * (distance / 1000.0);
    return fmax(RSSI_MIN, fmin(RSSI_MAX, rssi + noise));
}

// Calculate SNR based on antenna orientation and UAV position
float calculateSNR(float azimuth, float elevation, const UAV* uav, const EnvironmentalConditions* env) {
    float rssi = calculateRSSI(azimuth, elevation, uav, env);
    
    // Calculate noise floor with environmental factors
    float noiseFloor = NOISE_FLOOR;
    noiseFloor += env->interference * 10;
    noiseFloor += env->temperature * 0.1;
    noiseFloor += env->rain_intensity * 0.2;
    
    // Calculate SNR
    float snr = rssi - noiseFloor;
    
    // Clamp to realistic SNR ranges
    if(snr > 25) return 25 + randomFloat(0, 5);
    if(snr > 20) return 20 + randomFloat(0, 5);
    if(snr > 15) return 15 + randomFloat(0, 5);
    if(snr > 10) return 10 + randomFloat(0, 5);
    
    return randomFloat(0, 10);
}

// Calculate fitness based on RSSI and SNR
float calculateFitness(float rssi, float snr) {
    // Normalize RSSI and SNR to 0-1 range
    float rssi_norm = (rssi - RSSI_MIN) / (RSSI_MAX - RSSI_MIN);
    float snr_norm = (snr - SNR_MIN) / (SNR_MAX - SNR_MIN);

    // Weight factors
    const float rssi_weight = 0.6;
    const float snr_weight = 0.4;

    return rssi_weight * rssi_norm + snr_weight * snr_norm;
}

// Initialize particles
void initializeParticles(Particle* particles, float x_min, float x_max, float y_min, float y_max) {
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].position[0] = randomFloat(x_min, x_max);
        particles[i].position[1] = randomFloat(y_min, y_max);
        particles[i].velocity[0] = randomFloat(-5, 5);
        particles[i].velocity[1] = randomFloat(-3, 3);
        particles[i].fitness = -INFINITY;
        particles[i].bestFitness = -INFINITY;
        particles[i].stagnationCount = 0;
    }
}

// PSO main function
void runPSO(SystemState* state, EnvironmentalConditions* env, UAV* uav, int mode, float** rssiMap, float** snrMap) {
    Particle particles[NUM_PARTICLES];
    float w = INERTIA_MAX;

    // Determine search area boundaries
    float x_min, x_max, y_min, y_max;
    if (mode == 0) {
        x_min = AZIMUTH_MIN;
        x_max = AZIMUTH_MAX;
        y_min = ELEVATION_MIN;
        y_max = ELEVATION_MAX;
    } else {
        // Narrow search around best position
        x_min = fmax(AZIMUTH_MIN, state->globalBestPosition[0] - AZIMUTH_SEARCH_AREA / 2);
        x_max = fmin(AZIMUTH_MAX, state->globalBestPosition[0] + AZIMUTH_SEARCH_AREA / 2);
        y_min = fmax(ELEVATION_MIN, state->globalBestPosition[1] - ELEVATION_SEARCH_AREA / 2);
        y_max = fmin(ELEVATION_MAX, state->globalBestPosition[1] + ELEVATION_SEARCH_AREA / 2);
    }

    // Initialize particles
    initializeParticles(particles, x_min, x_max, y_min, y_max);

    // PSO iterations
    int max_iter = MAX_ITERATIONS;
    for (int iter = 0; iter < max_iter; iter++) {
        // Update PSO parameters
        updatePSOParameters(&w, iter, max_iter, particles);

        for (int i = 0; i < NUM_PARTICLES; i++) {
            // Discrete positions based on resolution
            int azIndex = (int)(wrapAngle360(particles[i].position[0]) / AZIMUTH_RESOLUTION);
            int elIndex = (int)(particles[i].position[1] / ELEVATION_RESOLUTION);

            // Ensure indices are within bounds
            if (azIndex >= AZIMUTH_POINTS) azIndex = AZIMUTH_POINTS - 1;
            if (elIndex >= ELEVATION_POINTS) elIndex = ELEVATION_POINTS - 1;
            if (azIndex < 0) azIndex = 0;
            if (elIndex < 0) elIndex = 0;

            // Retrieve RSSI and SNR from signal maps
            float rssi = rssiMap[azIndex][elIndex];
            float snr = snrMap[azIndex][elIndex];
            particles[i].fitness = calculateFitness(rssi, snr);

            // Update personal best
            if (particles[i].fitness > particles[i].bestFitness) {
                particles[i].bestFitness = particles[i].fitness;
                particles[i].bestPosition[0] = particles[i].position[0];
                particles[i].bestPosition[1] = particles[i].position[1];
                particles[i].stagnationCount = 0;
            } else {
                particles[i].stagnationCount++;
            }

            // Update global best
            if (particles[i].fitness > state->globalBestFitness) {
                state->globalBestFitness = particles[i].fitness;
                state->globalBestPosition[0] = particles[i].position[0];
                state->globalBestPosition[1] = particles[i].position[1];

                printf("\nNew Global Best Found!\n");
                printf("Position: Azimuth=%.2f°, Elevation=%.2f°\n",
                       state->globalBestPosition[0], state->globalBestPosition[1]);
                printf("Fitness: %.4f\n", state->globalBestFitness);
            }

            // Update velocity
            float r1 = randomFloat(0, 1);
            float r2 = randomFloat(0, 1);

            for (int d = 0; d < 2; d++) {
                particles[i].velocity[d] = w * particles[i].velocity[d] +
                    C1 * r1 * (particles[i].bestPosition[d] - particles[i].position[d]) +
                    C2 * r2 * (state->globalBestPosition[d] - particles[i].position[d]);
            }

            // Velocity clamping
            float v_max_x = (x_max - x_min) * 0.2;
            float v_max_y = (y_max - y_min) * 0.2;
            particles[i].velocity[0] = fmax(-v_max_x, fmin(v_max_x, particles[i].velocity[0]));
            particles[i].velocity[1] = fmax(-v_max_y, fmin(v_max_y, particles[i].velocity[1]));

            // Update position
            for (int d = 0; d < 2; d++) {
                particles[i].position[d] += particles[i].velocity[d];
            }

            // Wrap angles to valid ranges
            particles[i].position[0] = wrapAngle360(particles[i].position[0]);
            particles[i].position[1] = fmax(ELEVATION_MIN, fmin(ELEVATION_MAX, particles[i].position[1]));
        }

        // Increment total scans
        state->totalScans++;

        // Sleep to simulate time passing
        usleep(50000); // 0.005 seconds per iteration to simulate processing time
    }

    // After PSO, set target angles to global best found
    state->targetAngles[0] = state->globalBestPosition[0];
    state->targetAngles[1] = state->globalBestPosition[1];
}

// Generate random float between min and max
float randomFloat(float min, float max) {
    return min + (max - min) * ((float)rand() / RAND_MAX);
}

// Wrap angle to 0-360 degrees
float wrapAngle360(float angle) {
    while (angle < 0) angle += 360;
    while (angle >= 360) angle -= 360;
    return angle;
}

// Apply Kalman filter to smooth target angles
void applyKalmanFilter(SystemState* state) {
    for (int i = 0; i < 2; i++) {
        // Predict
        float predictedState = state->kalmanState[i];
        float predictedCovariance = state->kalmanCovariance[i] + KALMAN_Q;

        // Measurement
        float measurement = state->globalBestPosition[i];

        // Update
        float kalmanGain = predictedCovariance / (predictedCovariance + KALMAN_R);
        state->kalmanState[i] = predictedState + kalmanGain * (measurement - predictedState);
        state->kalmanCovariance[i] = (1 - kalmanGain) * predictedCovariance;

        // Update target angle
        state->targetAngles[i] = state->kalmanState[i];
    }
}

// Update servo angles towards target angles with movement constraints
void updateServoAngles(SystemState* state, float deltaTime) {
    // Hedef açılara yumuşak geçiş
    float maxAngularSpeed = MAX_ANGULAR_SPEED * deltaTime;
    
    for (int i = 0; i < 2; i++) {
        float targetAngle = state->targetAngles[i];
        float currentAngle = state->servoAngles[i];
        float angleDiff = (i == 0) ? 
            wrapAngle180(targetAngle - currentAngle) : 
            (targetAngle - currentAngle);
        
        float angleChange = fmin(fabs(angleDiff), maxAngularSpeed) * 
                          (angleDiff > 0 ? 1.0f : -1.0f);
        
        state->servoAngles[i] += angleChange;
        
        if (i == 0) {
            state->servoAngles[i] = wrapAngle360(state->servoAngles[i]);
        } else {
            state->servoAngles[i] = fmax(ELEVATION_MIN, 
                                       fmin(ELEVATION_MAX, state->servoAngles[i]));
        }
    }
}

// Print simulation status
void printSimulationStatus(const SystemState* state, const EnvironmentalConditions* env, const UAV* uav) {
    float rssi = calculateRSSI(state->servoAngles[0],
                               state->servoAngles[1], uav, env);
    float snr = calculateSNR(state->servoAngles[0],
                             state->servoAngles[1], uav, env);
    float fitness = calculateFitness(rssi, snr);

    printf("\nCurrent Signal Status:\n");
    printf("Antenna Orientation: Azimuth=%.2f°, Elevation=%.2f°\n",
           state->servoAngles[0], state->servoAngles[1]);
    printf("RSSI: %.2f dBm\n", rssi);
    printf("SNR: %.2f dB\n", snr);
    printf("Signal Quality (Fitness): %.4f\n", fitness);

    // Print environmental conditions
    printf("Environmental Conditions:\n");
    printf("Temperature: %.2f°C, Humidity: %.2f%%, Interference: %.2f\n",
           env->temperature, env->humidity, env->interference);
    printf("Rain Intensity: %.2f mm/h, Wind Speed: %.2f m/s\n",
           env->rain_intensity, env->wind_speed);

    // Print UAV position
    printf("UAV Position: X=%.2f m, Y=%.2f m, Altitude=%.2f m\n",
           uav->position[0], uav->position[1], uav->position[2]);

    // Calculate actual azimuth and elevation to UAV
    float actualAzimuth, actualElevation;
    calculateAzimuthElevationToUAV(uav, &actualAzimuth, &actualElevation);
    printf("Actual UAV Direction: Azimuth=%.2f°, Elevation=%.2f°\n", actualAzimuth, actualElevation);
}

// Update PSO parameters based on swarm diversity
void updatePSOParameters(float* w, int iteration, int max_iterations, Particle* particles) {
    // Calculate swarm diversity
    float positionMean[2] = {0.0, 0.0};
    for (int i = 0; i < NUM_PARTICLES; i++) {
        positionMean[0] += particles[i].position[0];
        positionMean[1] += particles[i].position[1];
    }
    positionMean[0] /= NUM_PARTICLES;
    positionMean[1] /= NUM_PARTICLES;

    float diversity = 0.0;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        diversity += pow(particles[i].position[0] - positionMean[0], 2) +
                     pow(particles[i].position[1] - positionMean[1], 2);
    }
    diversity = sqrt(diversity / NUM_PARTICLES);

    // Adjust inertia weight based on diversity
    *w = INERTIA_MIN + (INERTIA_MAX - INERTIA_MIN) * (diversity / 100.0);
    *w = fmax(INERTIA_MIN, fmin(INERTIA_MAX, *w));
}

// Update signal map based on UAV position
void updateSignalMap(float** rssiMap, float** snrMap, const UAV* uav, const EnvironmentalConditions* env) {
    for (int azimuth = AZIMUTH_MIN; azimuth < AZIMUTH_MAX; azimuth += AZIMUTH_RESOLUTION) {
        for (int elevation = ELEVATION_MIN; elevation < ELEVATION_MAX; elevation += ELEVATION_RESOLUTION) {
            int azIndex = azimuth / AZIMUTH_RESOLUTION;
            int elIndex = elevation / ELEVATION_RESOLUTION;
            float rssi = calculateRSSI(azimuth, elevation, uav, env);
            float snr = calculateSNR(azimuth, elevation, uav, env);
            rssiMap[azIndex][elIndex] = rssi;
            snrMap[azIndex][elIndex] = snr;
        }
    }
}

// Allocate memory for signal maps
bool allocateSignalMaps(float*** rssiMap, float*** snrMap) {
    size_t totalSize = AZIMUTH_POINTS * ELEVATION_POINTS;
    
    signalMaps.rssiMap = (float**)malloc(AZIMUTH_POINTS * sizeof(float*));
    signalMaps.snrMap = (float**)malloc(AZIMUTH_POINTS * sizeof(float*));
    signalMaps.rssiData = (float*)malloc(totalSize * sizeof(float));
    signalMaps.snrData = (float*)malloc(totalSize * sizeof(float));
    
    CHECK_MALLOC(signalMaps.rssiMap);
    CHECK_MALLOC(signalMaps.snrMap);
    CHECK_MALLOC(signalMaps.rssiData);
    CHECK_MALLOC(signalMaps.snrData);

    for (int i = 0; i < AZIMUTH_POINTS; i++) {
        signalMaps.rssiMap[i] = signalMaps.rssiData + (i * ELEVATION_POINTS);
        signalMaps.snrMap[i] = signalMaps.snrData + (i * ELEVATION_POINTS);
    }

    *rssiMap = signalMaps.rssiMap;
    *snrMap = signalMaps.snrMap;
    return true;
}

// Free memory of signal maps
void freeSignalMaps(float** rssiMap, float** snrMap) {
    if (signalMaps.rssiData) free(signalMaps.rssiData);
    if (signalMaps.snrData) free(signalMaps.snrData);
    if (signalMaps.rssiMap) free(signalMaps.rssiMap);
    if (signalMaps.snrMap) free(signalMaps.snrMap);
    memset(&signalMaps, 0, sizeof(SignalMaps));
}

float calculateDetailedAntennaGain(float azimuthError, float elevationError) {
    // 3D antenna pattern based on realistic parabolic antenna characteristics
    const float maxGain = 25.0; // dB
    const float beamwidth3dB = 15.0; // degrees
    const float frontToBack = 25.0; // dB
    const float sidelobeLevel = -18.0; // dB

    // Calculate normalized angular distance from boresight
    float normalizedError = sqrt(pow(azimuthError/beamwidth3dB, 2) + 
                               pow(elevationError/beamwidth3dB, 2));

    // Main beam approximation using modified Gaussian pattern
    if(normalizedError <= 1.0) {
        return maxGain * exp(-2.77 * pow(normalizedError, 2));
    }
    // Side lobe region
    else if(normalizedError <= 2.0) {
        return maxGain + sidelobeLevel;
    }
    // Back lobe region
    else {
        return maxGain - frontToBack;
    }
}

float calculateWaterVaporAttenuation(float humidity, float temperature) {
    // Based on ITU-R P.676-12 simplified model
    const float freq = 2.4; // GHz
    const float pressure = 1013.25; // Standard atmospheric pressure (hPa)
    
    // Calculate water vapor density (g/m³)
    float esat = 6.1121 * exp((17.502 * temperature)/(temperature + 240.97));
    float wvd = humidity/100.0 * esat * 217.0/(temperature + 273.15);
    
    // Simplified water vapor attenuation at 2.4 GHz
    float attenuation = 0.0007 * wvd * (300.0/(temperature + 273.15)) * 
                       exp(-2.0 * (freq - 2.4));
    
    return fmax(0.0, attenuation);
}

float calculateRicianFading(float K, float multipathFactor) {
    // Rice distribution for multipath fading
    float directPower = pow(10.0, K/10.0);
    float scatteredPower = 1.0;
    
    // Generate Rice distributed random variable
    float x = randomFloat(-1, 1) * sqrt(scatteredPower/2);
    float y = randomFloat(-1, 1) * sqrt(scatteredPower/2);
    float direct = sqrt(directPower);
    
    float magnitude = sqrt(pow(x + direct, 2) + pow(y, 2));
    return -20 * log10(magnitude) * multipathFactor;
}

float calculateUrbanLoss(float terrainFactor, float distance) {
    // Modified COST231-Walfisch-Ikegami model
    const float baseHeight = 30.0; // meters
    const float freq = 2400.0; // MHz
    
    float pathLoss = 42.6 + 26 * log10(distance/1000.0) + 20 * log10(freq/1000.0);
    float buildingLoss = terrainFactor * 20; // Additional loss due to buildings
    float diffractionLoss = terrainFactor * 10 * log10(distance/100.0);
    
    return pathLoss + buildingLoss + diffractionLoss;
}

float calculateDopplerShift(const UAV* uav, float lambda) {
    // Calculate relative velocity vector between UAV and antenna
    float vr = -(uav->velocity[0] * uav->position[0] + 
                 uav->velocity[1] * uav->position[1] + 
                 uav->velocity[2] * uav->position[2]) / 
                sqrt(pow(uav->position[0], 2) + 
                     pow(uav->position[1], 2) + 
                     pow(uav->position[2], 2));
    
    // Calculate Doppler shift (Hz)
    float c = 3e8; // Speed of light (m/s)
    float freq = c/lambda;
    float dopplerFreq = freq * vr/c;
    
    // Convert to dB impact on signal
    return 20 * log10(fabs(dopplerFreq)/1000.0 + 1.0);
}

float calculateInterference(float interferenceLevel) {
    // Model various interference sources
    const float backgroundNoise = -100.0; // dBm
    const float maxInterference = -50.0; // dBm
    
    // Calculate composite interference
    float wifiInterference = interferenceLevel * -10.0;
    float bluetoothInterference = interferenceLevel * -5.0;
    float otherDevices = interferenceLevel * -15.0;
    
    // Combine interference sources (logarithmic addition)
    float totalInterference = 10 * log10(
        pow(10, wifiInterference/10.0) +
        pow(10, bluetoothInterference/10.0) +
        pow(10, otherDevices/10.0)
    );
    
    return fmax(backgroundNoise, fmin(maxInterference, totalInterference));
}

float calculateSystemNoise(void) {
    // System noise temperature calculation
    const float T0 = 290.0; // Reference temperature (K)
    const float NF = 3.0;   // Noise Figure (dB)
    const float B = 20e6;   // Bandwidth (Hz)
    const float k = 1.38e-23; // Boltzmann constant
    
    // Calculate noise floor
    float noiseFigure = pow(10.0, NF/10.0);
    float systemTemp = T0 * (noiseFigure - 1);
    float thermalNoise = 10 * log10(k * (T0 + systemTemp) * B * 1000); // Convert to dBm
    
    // Add random fluctuation
    float fluctuation = randomFloat(-0.5, 0.5);
    
    return thermalNoise + fluctuation;
}
// Atmosferik etkilerin uygulanması
void applyAtmosphericEffects(UAV* uav, const EnvironmentalConditions* env) {
    // Türbülans etkisi
    for (int i = 0; i < 3; i++) {
        uav->velocity[i] += env->turbulenceIntensity * randomFloat(-1.0f, 1.0f);
    }
    
    // Hava yoğunluğu etkisi
    float densityFactor = env->airDensity / 1.225f; // Deniz seviyesi yoğunluğuna göre normalize
    for (int i = 0; i < 3; i++) {
        uav->acceleration[i] *= densityFactor;
    }
}

// Motor kuvvetlerinin güncellenmesi
void updateEngineForces(UAV* uav) {
    const float maxThrust = 20.0f; // Newtons
    const float batteryEfficiency = uav->batteryLevel / 100.0f;
    
    for (int i = 0; i < 4; i++) {
        uav->enginePower[i] = uav->throttle * maxThrust * batteryEfficiency;
    }
}

// Fizik motorunun güncellenmesi
void updatePhysics(UAV* uav, float deltaTime) {
    // Yerçekimi etkisi
    const float gravity = 9.81f;
    uav->acceleration[2] -= gravity;
    
    // Euler integrasyonu
    for (int i = 0; i < 3; i++) {
        uav->velocity[i] += uav->acceleration[i] * deltaTime;
        uav->position[i] += uav->velocity[i] * deltaTime;
    }
    
    // Jiroskop verilerini güncelle
    uav->gyroscope[0] = uav->roll;
    uav->gyroscope[1] = uav->pitch;
    uav->gyroscope[2] = uav->heading;
}

// Mesafe hesaplama
float calculateDistance(const UAV* uav) {
    return sqrt(pow(uav->position[0], 2) + 
               pow(uav->position[1], 2) + 
               pow(uav->position[2], 2));
}

// Yol kaybı hesaplama
float calculatePathLoss(float distance, const EnvironmentalConditions* env) {
    const float freq = 2.4e9; // 2.4 GHz
    const float c = 3e8; // Speed of light
    float lambda = c / freq;
    
    // Free space path loss
    float fspl = 20 * log10(distance) + 20 * log10(freq) - 147.55;
    
    // Additional losses
    float environmentalLoss = env->rain_intensity * 0.1f + 
                            env->humidity * 0.05f +
                            env->temperature * 0.02f;
    
    return fspl + environmentalLoss;
}

// Atmosferik kayıpların hesaplanması
float calculateAtmosphericLoss(const EnvironmentalConditions* env, float distance) {
    float atmosphericLoss = env->atmospheric_loss * distance / 1000.0;
    float temperatureFactor = 0.01 * fabs(env->temperature - 15.0); // 15°C referans
    float humidityFactor = 0.005 * env->humidity;
    float pressureFactor = 0.001 * fabs(env->atmosphericPressure - 1013.25);
    
    return atmosphericLoss + temperatureFactor + humidityFactor + pressureFactor;
}

// Yönsel kayıpların hesaplanması
float calculateDirectionalLoss(const UAV* uav) {
    float horizontalAngle = atan2(uav->velocity[1], uav->velocity[0]);
    float verticalAngle = atan2(uav->velocity[2], 
                               sqrt(pow(uav->velocity[0], 2) + pow(uav->velocity[1], 2)));
    
    // Anten yönelim kaybı
    float antennaLoss = 0.1 * (fabs(horizontalAngle) + fabs(verticalAngle));
    
    // Platformun yönelim etkisi
    float platformEffect = 0.05 * (fabs(uav->roll) + fabs(uav->pitch));
    
    return antennaLoss + platformEffect;
}

// Diğer kayıpların hesaplanması
float calculateOtherLosses(const EnvironmentalConditions* env) {
    float cloudLoss = env->cloudCover * 0.5;
    float solarEffect = env->solarRadiation * 0.001;
    float magneticEffect = env->magneticInterference * 1000.0;
    
    return cloudLoss + solarEffect + magneticEffect;
}

// Kalman filtresi durum tahmini
void predictState(KalmanFilter* kf, const UAV* uav) {
    // Durum geçiş matrisi
    const float dt = 0.1; // Zaman adımı
    
    // Pozisyon ve hız tahminleri
    for(int i = 0; i < 3; i++) {
        kf->state[i] = kf->state[i] + dt * kf->state[i+3];
        kf->state[i+3] = kf->state[i+3];
    }
    
    // Kovaryans matrisinin güncellenmesi
    // Basitleştirilmiş güncelleme
    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
            kf->covariance[i][j] += kf->processNoise[i][j];
        }
    }
}

// Kalman filtresi ölçüm güncellemesi
void updateMeasurement(KalmanFilter* kf, const float* measurements) {
    const float R = 0.1; // Ölçüm gürültüsü
    
    // Kalman kazancı hesaplama
    float K[6][3];
    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 3; j++) {
            K[i][j] = kf->covariance[i][j] / (kf->covariance[j][j] + R);
        }
    }
    
    // Durum güncellemesi
    for(int i = 0; i < 6; i++) {
        float innovation = 0;
        for(int j = 0; j < 3; j++) {
            innovation += K[i][j] * (measurements[j] - kf->state[j]);
        }
        kf->state[i] += innovation;
    }
}

// Kalman filtresi durum güncellemesi
void updateState(KalmanFilter* kf, UAV* uav) {
    // UAV durumunu güncelle
    for(int i = 0; i < 3; i++) {
        uav->position[i] = kf->state[i];
        uav->velocity[i] = kf->state[i+3];
    }
}

// PSO parçacık sayısı optimizasyonu
int calculateOptimalParticleCount(const SystemState* state) {
    // Arama alanının büyüklüğüne göre parçacık sayısını ayarla
    float searchArea;
    if(state->searchMode == 0) {
        searchArea = AZIMUTH_MAX * ELEVATION_MAX;
    } else {
        searchArea = AZIMUTH_SEARCH_AREA * ELEVATION_SEARCH_AREA;
    }
    
    // Minimum 4, maksimum 20 parçacık
    return (int)fmin(20, fmax(4, sqrt(searchArea) / 10));
}

// Arama alanı optimizasyonu
void optimizeSearchSpace(SystemState* state, const UAV* uav) {
    float azimuth, elevation;
    calculateAzimuthElevationToUAV(uav, &azimuth, &elevation);
    
    // Tahmin edilen konuma göre arama alanını ayarla
    if(state->searchMode == 1) {
        state->targetAngles[0] = azimuth;
        state->targetAngles[1] = elevation;
    }
}

// Parçacıkların güncellenmesi
void updateParticles(SystemState* state, const FlightMatrix* matrix) {
    // Matrix boyutlarına göre parçacıkları güncelle
    float scaleFactor = matrix->cellSize;
    state->timeSinceLastUpdate += 0.1;
    
    // PSO parçacıklarını güncelle
    // Bu fonksiyon runPSO içinde zaten implementeydi
}

// Global en iyi konumun güncellenmesi
void updateGlobalBest(SystemState* state) {
    // Önceki en iyi değeri sakla
    state->previousBestFitness = state->globalBestFitness;
    
    // Yeni en iyi değer state->globalBestFitness içinde zaten güncelleniyor
}

// Yakınsama kontrolü
bool checkConvergence(const SystemState* state) {
    const float convergenceThreshold = 0.001;
    
    // Fitness değişimi yakınsama eşiğinden küçükse
    if(fabs(state->globalBestFitness - state->previousBestFitness) < convergenceThreshold) {
        return true;
    }
    return false;
}

// ...remaining new function implementations...

// UAV uçuş parametrelerini güncelle
#define TAKEOFF_INITIAL_ALT 0.0f    // Başlangıç irtifası (m)
#define TAKEOFF_SPEED 2.0f          // Kalkış hızı (m/s)
#define CRUISE_SPEED 12.0f          // Seyir hızı (m/s)
#define MAX_CLIMB_RATE 3.0f         // Maksimum tırmanma hızı (m/s)
#define MAX_DESCENT_RATE -2.0f      // Maksimum alçalma hızı (m/s)
#define CIRCLE_RADIUS 150.0f        // Dönüş yarıçapı (m)
#define CRUISE_ALTITUDE 100.0f      // Seyir irtifası (m)
#define TAKEOFF_PITCH 15.0f         // Kalkış yunuslama açısı (derece)
#define CRUISE_PITCH 5.0f           // Seyir yunuslama açısı (derece)

// Kalkış fonksiyonunu güncelle
void performTakeoff(UAV* uav, float deltaTime) {
    const float transitionAlt = 30.0f; // Geçiş irtifası
    
    if (uav->position[2] < transitionAlt) {
        // İlk tırmanma fazı
        uav->throttle = 0.9f;
        uav->pitch = TAKEOFF_PITCH;
        uav->velocity[2] = TAKEOFF_SPEED;
        
        // Yavaşça ileri hareketi başlat
        uav->velocity[0] = TAKEOFF_SPEED * 0.5f;
        uav->velocity[1] = 0.0f;
    } 
    else if (uav->position[2] < CRUISE_ALTITUDE) {
        // Ana tırmanma fazı
        float altitudeRemaining = CRUISE_ALTITUDE - uav->position[2];
        float climbRate = fmin(MAX_CLIMB_RATE, altitudeRemaining * 0.1f);
        
        uav->throttle = 0.7f;
        uav->pitch = TAKEOFF_PITCH * (altitudeRemaining / CRUISE_ALTITUDE);
        uav->velocity[2] = climbRate;
        
        // Kademeli olarak seyir hızına geç
        float speedRatio = (uav->position[2] - transitionAlt) / (CRUISE_ALTITUDE - transitionAlt);
        uav->velocity[0] = TAKEOFF_SPEED + (CRUISE_SPEED - TAKEOFF_SPEED) * speedRatio;
    } 
    else {
        // Seyir fazına geç
        uav->flightState = CRUISE;
        uav->throttle = 0.6f;
        uav->pitch = CRUISE_PITCH;
        initializeCircularPath(uav);
    }
    
    // Pozisyon güncelleme
    for(int i = 0; i < 3; i++) {
        uav->position[i] += uav->velocity[i] * deltaTime;
    }
}

// Seyir fonksiyonunu güncelle
void performCruise(UAV* uav, float deltaTime, const EnvironmentalConditions* env) {
    const float CIRCLE_PERIOD = 2.0f * M_PI * CIRCLE_RADIUS / CRUISE_SPEED;
    float omega = 2.0f * M_PI / CIRCLE_PERIOD;
    
    // Dairesel yörünge hesaplama
    float targetX = CIRCLE_RADIUS * cos(omega * uav->pathTime);
    float targetY = CIRCLE_RADIUS * sin(omega * uav->pathTime);
    float targetZ = CRUISE_ALTITUDE;
    
    // Pozisyon hatası hesaplama
    float errorX = targetX - uav->position[0];
    float errorY = targetY - uav->position[1];
    float errorZ = targetZ - uav->position[2];
    
    // Hedef hızları hesapla
    float targetVx = -CIRCLE_RADIUS * omega * sin(omega * uav->pathTime);
    float targetVy = CIRCLE_RADIUS * omega * cos(omega * uav->pathTime);
    
    // Yumuşak hız geçişleri
    float speedTransitionRate = 0.1f;
    uav->velocity[0] += (targetVx - uav->velocity[0]) * speedTransitionRate;
    uav->velocity[1] += (targetVy - uav->velocity[1]) * speedTransitionRate;
    uav->velocity[2] = fmax(MAX_DESCENT_RATE, fmin(MAX_CLIMB_RATE, errorZ * 0.5f));
    
    // Yunuslama ve yatış açılarını güncelle
    float targetRoll = atan2(pow(uav->velocity[0], 2) + pow(uav->velocity[1], 2), 
                            9.81f * CIRCLE_RADIUS) * 180.0f / M_PI;
    uav->roll = targetRoll;
    uav->pitch = CRUISE_PITCH + (errorZ > 0 ? 2.0f : -2.0f);
    
    // Rüzgar etkisini uygula
    uav->velocity[0] += env->wind_speed * 0.1f * cos(env->time_variation);
    uav->velocity[1] += env->wind_speed * 0.1f * sin(env->time_variation);
    
    // Pozisyon güncelleme
    for(int i = 0; i < 3; i++) {
        uav->position[i] += uav->velocity[i] * deltaTime;
    }
    
    uav->pathTime += deltaTime;
}

// Dairesel yörünge başlatma fonksiyonu
void initializeCircularPath(UAV* uav) {
    // Başlangıç pozisyonu ayarla
    uav->pathTime = 0.0f;
    uav->position[0] = CIRCLE_RADIUS; // X başlangıç pozisyonu
    uav->position[1] = 0.0f;          // Y başlangıç pozisyonu
    uav->position[2] = CRUISE_ALTITUDE; // Seyir irtifası
    
    // Başlangıç hızını ayarla
    const float CIRCLE_PERIOD = 2.0f * M_PI * CIRCLE_RADIUS / CRUISE_SPEED;
    float omega = 2.0f * M_PI / CIRCLE_PERIOD;
    
    uav->velocity[0] = 0.0f;                    // Başlangıçta X yönünde hız
    uav->velocity[1] = CRUISE_SPEED;            // Başlangıçta Y yönünde hız
    uav->velocity[2] = 0.0f;                    // Başlangıçta dikey hız
    
    // Başlangıç yönelimini ayarla
    uav->heading = 0.0f;                        // Başlangıç heading açısı
    uav->pitch = CRUISE_PITCH;                  // Seyir yunuslama açısı
    uav->roll = atan2(pow(CRUISE_SPEED, 2),    // Dönüş için gerekli yatış açısı
                      9.81f * CIRCLE_RADIUS) * 180.0f / M_PI;
    
    // Motor gücünü ayarla
    uav->throttle = 0.6f;                       // Seyir gücü
    
    // Jiroskop verilerini sıfırla
    for(int i = 0; i < 3; i++) {
        uav->gyroscope[i] = 0.0f;
        uav->accelerometer[i] = 0.0f;
    }
}

// ...existing code until UAV parameters...

// Güncellenmiş UAV parametreleri
#define UAV_MIN_SPEED 20.0f         // Minimum hız (m/s)
#define UAV_MAX_SPEED 30.0f         // Maksimum hız (m/s)
#define UAV_MIN_ALTITUDE 100.0f     // Minimum irtifa (m)
#define UAV_MAX_ALTITUDE 300.0f     // Maksimum irtifa (m)
#define UAV_CLIMB_RATE 5.0f         // Tırmanma hızı (m/s)
#define UAV_TURN_RATE 0.2f          // Dönüş hızı (rad/s)
#define CIRCLE_MIN_RADIUS 100.0f    // Minimum dönüş yarıçapı (m)
#define CIRCLE_MAX_RADIUS 300.0f    // Maksimum dönüş yarıçapı (m)

// Sinyal parametreleri
#define RSSI_DISTANCE_FACTOR 20.0f  // Mesafe bazlı zayıflama (dB/dekad)
#define SNR_BASE_LEVEL 30.0f        // Temel SNR seviyesi
#define QUALITY_RSSI_WEIGHT 0.7f    // RSSI ağırlığı
#define QUALITY_SNR_WEIGHT 0.3f     // SNR ağırlığı

// Kalman filtresi parametreleri
#define KF_POS_VARIANCE 0.1f        // Pozisyon ölçüm varyansı
#define KF_VEL_VARIANCE 0.5f        // Hız ölçüm varyansı
#define KF_PROCESS_NOISE 0.01f      // Süreç gürültüsü

// UAV pozisyon güncelleme fonksiyonunu güncelle
void updateUAVPosition(UAV* uav, const EnvironmentalConditions* env, float deltaTime) {
    // Rastgele hareket parametreleri
    static float currentRadius = 200.0f;
    static float currentAltitude = 150.0f;
    static float timeAccumulator = 0.0f;

    // Her 10 saniyede bir hareket parametrelerini güncelle
    timeAccumulator += deltaTime;
    if(timeAccumulator >= 10.0f) {
        currentRadius = randomFloat(CIRCLE_MIN_RADIUS, CIRCLE_MAX_RADIUS);
        currentAltitude = randomFloat(UAV_MIN_ALTITUDE, UAV_MAX_ALTITUDE);
        timeAccumulator = 0.0f;
    }

    // Dairesel hareket
    float omega = UAV_TURN_RATE * (1.0f + 0.2f * sin(timeAccumulator * 0.1f));
    float targetX = currentRadius * cos(omega * uav->pathTime);
    float targetY = currentRadius * sin(omega * uav->pathTime);
    float targetZ = currentAltitude + 20.0f * sin(timeAccumulator * 0.05f);

    // Hedef hızları hesapla
    float targetVx = -currentRadius * omega * sin(omega * uav->pathTime);
    float targetVy = currentRadius * omega * cos(omega * uav->pathTime);
    float targetVz = (targetZ - uav->position[2]) * 0.5f;

    // Hız sınırlamaları ve yumuşatma
    for(int i = 0; i < 3; i++) {
        float velocityDiff = ((i == 0) ? targetVx : (i == 1) ? targetVy : targetVz) - uav->velocity[i];
        uav->velocity[i] += velocityDiff * ACCELERATION_SMOOTHING;
    }

    // Rüzgar ve türbülans etkileri
    applyWindEffects(uav, env, deltaTime);

    // Pozisyon güncelleme
    for(int i = 0; i < 3; i++) {
        uav->position[i] += uav->velocity[i] * deltaTime;
    }

    // Açısal değerleri güncelle
    updateAttitudeAngles(uav);

    uav->pathTime += deltaTime;
}


// Kalman filtresi uygulamasını güncelle
void applyExtendedKalmanFilter(KalmanFilter* kf, UAV* uav, float* measurements) {
    // Durum tahmini
    float dt = 0.1f;
    float F[6][6] = {{1,0,0,dt,0,0},
                     {0,1,0,0,dt,0},
                     {0,0,1,0,0,dt},
                     {0,0,0,1,0,0},
                     {0,0,0,0,1,0},
                     {0,0,0,0,0,1}};
    
    // Durum tahminini uygula
    predictKalmanState(kf, F);
    
    // Ölçüm güncellemesi
    updateKalmanMeasurement(kf, measurements);
    
    // Filtrelenmiş durumu UAV'ye uygula
    applyKalmanToUAV(kf, uav);
}

// Yeni yardımcı fonksiyonlar
void applyWindEffects(UAV* uav, const EnvironmentalConditions* env, float deltaTime) {
    // Rüzgar etkisi
    for(int i = 0; i < 3; i++) {
        uav->windEffect[i] = env->wind_speed * randomFloat(-1.0f, 1.0f) * deltaTime;
        uav->velocity[i] += uav->windEffect[i];
    }
    
    // Türbülans etkisi
    float turbulence = env->turbulenceIntensity * randomFloat(-1.0f, 1.0f);
    for(int i = 0; i < 3; i++) {
        uav->velocity[i] += turbulence * deltaTime;
    }
}

void updateAttitudeAngles(UAV* uav) {
    // Heading güncelleme (yaw)
    uav->heading = atan2(uav->velocity[1], uav->velocity[0]);
    
    // Pitch güncelleme
    float horizontalSpeed = sqrt(pow(uav->velocity[0], 2) + pow(uav->velocity[1], 2));
    uav->pitch = atan2(uav->velocity[2], horizontalSpeed);
    
    // Roll güncelleme (dönüş için gerekli bank açısı)
    float turnRadius = horizontalSpeed / uav->turnRate;
    float centralAcceleration = pow(horizontalSpeed, 2) / turnRadius;
    uav->roll = atan2(centralAcceleration, 9.81f);
}

float calculateEnvironmentalLoss(const EnvironmentalConditions* env, float distance) {
    float loss = 0.0f;
    
    // Mesafe bazlı kayıplar
    loss += 0.1f * distance / 1000.0f; // Her km için 0.1 dB
    
    // Hava koşulları kayıpları
    loss += env->rain_intensity * 0.2f;  // Yağmur etkisi
    loss += env->cloudCover * 0.5f;      // Bulut etkisi
    loss += env->humidity * 0.01f;       // Nem etkisi
    
    // Atmosferik kayıplar
    float atmosphericLoss = env->atmospheric_loss * distance / 1000.0f;
    loss += atmosphericLoss;
    
    return loss;
}

// Kalman filtresi yardımcı fonksiyonları
void predictKalmanState(KalmanFilter* kf, float F[6][6]) {
    // Durum tahmini
    float tempState[6] = {0};
    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
            tempState[i] += F[i][j] * kf->state[j];
        }
    }
    memcpy(kf->state, tempState, sizeof(kf->state));
}

void updateKalmanMeasurement(KalmanFilter* kf, const float* measurements) {
    const float R = KF_POS_VARIANCE;
    for(int i = 0; i < 3; i++) {
        float innovation = measurements[i] - kf->state[i];
        float gain = kf->covariance[i][i] / (kf->covariance[i][i] + R);
        kf->state[i] += gain * innovation;
    }
}

void applyKalmanToUAV(KalmanFilter* kf, UAV* uav) {
    // Filtrelenmiş durumu UAV'ye uygula
    for(int i = 0; i < 3; i++) {
        uav->position[i] = kf->state[i];
        uav->velocity[i] = kf->state[i + 3];
    }
}

float wrapAngle180(float angle) {
    angle = fmod(angle + 180, 360);
    if (angle < 0) angle += 360;
    return angle - 180;
}

float calculateAntennaPatternLoss(float azimuthError, float elevationError) {
    const float beamwidth = 30.0f; // derece
    float normalizedError = sqrt(pow(azimuthError/beamwidth, 2) + 
                               pow(elevationError/beamwidth, 2));
    
    if (normalizedError <= 1.0f) {
        return 0.0f;
    } else {
        return 20.0f * log10(normalizedError); // dB cinsinden kayıp
    }
}
