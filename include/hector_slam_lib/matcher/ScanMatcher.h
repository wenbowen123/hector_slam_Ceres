#ifndef _scanmatcher_h__
#define _scanmatcher_h__

#include <fstream>
#include <Eigen/Geometry>
#include "../scan/DataPointContainer.h"
#include "../util/UtilFunctions.h"

#include "../util/DrawInterface.h"
#include "../util/HectorDebugInfoInterface.h"
#include <fstream>
#include <limits>
#include <ceres/ceres.h>


namespace hectorslam{
using namespace ceres;


template<typename ConcreteOccGridMapUtil>
class getResidual : public ceres::SizedCostFunction<1,3>
{
public:
    ConcreteOccGridMapUtil* occ;
    Eigen::Vector2f currPoint;


    getResidual(ConcreteOccGridMapUtil* occ, Eigen::Vector2f currPoint)
    {
        this->occ = occ;
        this->currPoint = currPoint;
    }
    virtual ~getResidual() {}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const
    {
        Eigen::Matrix<double, 3, 1> pose1(parameters[0][0],parameters[0][1],parameters[0][2]);
        Eigen::Vector3f pose = pose1.cast<float>();
        Eigen::Affine2f transform(occ->getTransformForState(pose)); // transform: rotation->translation


        float sinRot = std::sin(pose[2]);
        float cosRot = std::cos(pose[2]);


        Eigen::Vector3f transformedPointData(occ->interpMapValueWithDerivatives(transform * currPoint));  /// {M,dM/dx,dM/dy}

        float funVal = 1.0f - transformedPointData[0];
        //      float weight=util::WeightValue(funVal);
        float weight=1.0;

        residuals[0] = static_cast<double>(funVal);
        if (jacobians == NULL )
        {
          return true;
        }

        double rotDeriv = ((-sinRot * currPoint.x() - cosRot * currPoint.y()) * transformedPointData[1] + (cosRot * currPoint.x() - sinRot * currPoint.y()) * transformedPointData[2]);

        if (jacobians[0]!=NULL)
        {
            jacobians[0][0] = static_cast<double>(-transformedPointData[1]);
            jacobians[0][1] = static_cast<double>(-transformedPointData[2]);
            jacobians[0][2] = static_cast<double>(-rotDeriv);
        }


        return true;
    }
};


//template<typename ConcreteOccGridMapUtil>
//class GetResidualAutoDiff
//{
//public:
//    ConcreteOccGridMapUtil* occ;
//    Eigen::Vector2f currPoint;


//    GetResidualAutoDiff(ConcreteOccGridMapUtil* occ, Eigen::Vector2f currPoint)
//    {
//        this->occ = occ;
//        this->currPoint = currPoint;
//    }
//    virtual ~GetResidualAutoDiff() {}
//    template <typename T>
//      bool operator()(const T* parameters, T* residuals) const
//    {
//        Eigen::Matrix<T, 3, 1> pose(parameters[0],parameters[1],parameters[2]);
//        Eigen::Affine2f transform = occ->getTransformForState(pose);  // transform: rotation->translation


//        Eigen::Vector3f tmp1 = occ->interpMapValueWithDerivatives(  transform * currPoint);

//        Eigen::Matrix<T, 3, 1> transformedPointData(tmp1);  /// {M,dM/dx,dM/dy}

//        T funVal = T(1) - transformedPointData[0];

//        residuals[0] = funVal;

//        return true;
//    }
//};


template<typename ConcreteOccGridMapUtil>
class ScanMatcher
{
public:

    ScanMatcher(DrawInterface* drawInterfaceIn = 0, HectorDebugInfoInterface* debugInterfaceIn = 0)
        : drawInterface(drawInterfaceIn)
        , debugInterface(debugInterfaceIn)
    {
        error=std::numeric_limits<float>::max();
        error_last=std::numeric_limits<float>::max();
        lambda_max=1000;
        lambda_min=1e-4;
    }

    ~ScanMatcher()
    {}

    Eigen::Vector3f matchData(const Eigen::Vector3f& beginEstimateWorld, ConcreteOccGridMapUtil& gridMapUtil, const DataContainer& dataContainer, Eigen::Matrix3f& covMatrix, int maxIterations)
    {
        error=std::numeric_limits<float>::max();
        error_last=std::numeric_limits<float>::max();

        begin_estimation=1;
        if (drawInterface)
        {
            drawInterface->setScale(0.05f);
            drawInterface->setColor(0.0f,1.0f, 0.0f);
            drawInterface->drawArrow(beginEstimateWorld);

            Eigen::Vector3f beginEstimateMap(gridMapUtil.getMapCoordsPose(beginEstimateWorld));

            drawScan(beginEstimateMap, gridMapUtil, dataContainer);

            drawInterface->setColor(1.0,0.0,0.0);
        }

        if (dataContainer.getSize() != 0)
        {

            Eigen::Vector3f beginEstimateMap(gridMapUtil.getMapCoordsPose(beginEstimateWorld));

            Eigen::Vector3f estimate(beginEstimateMap);

            int method=1; /// 0 for Gauss, 1 for LM


            estimateTransformationLogLh(estimate, gridMapUtil, dataContainer,method);

            //      int numIter = maxIterations;

//            std::ofstream file2("/home/wbw/search.txt",std::ofstream::app);
//            file2<<"=\n";
//            file2.close();

//            std::ofstream file1("/home/wbw/error_original.txt",std::ofstream::app);
//            file1<<err_original<<"\n";
//            file1.close();


//            std::ofstream file("/home/wbw/ite.txt",std::ofstream::app);
//            file<<iter_count<<"\n";
//            file.close();


            if (drawInterface){
                drawInterface->setColor(0.0,0.0,1.0);
                drawScan(estimate, gridMapUtil, dataContainer);
            }

            estimate[2] = util::normalize_angle(estimate[2]);

            covMatrix = Eigen::Matrix3f::Zero();


//            covMatrix = H;

            return gridMapUtil.getWorldCoordsPose(estimate);
        }

        return beginEstimateWorld;
    }

protected:

    void estimateTransformationLogLh(Eigen::Vector3f& estimate, ConcreteOccGridMapUtil& gridMapUtil, const DataContainer& dataPoints,int method)
    {
        int size = dataPoints.getSize();
        ceres::Problem problem;
        double *parameter;
        Eigen::Vector3d estimate1 = estimate.cast<double>();
        parameter = estimate1.data();




        for (int i=0; i<size; i++)
        {
            const Eigen::Vector2f& currPoint (dataPoints.getVecEntry(i));
            ceres::CostFunction* cost_function = new getResidual< OccGridMapUtilConfig<GridMap> >(&gridMapUtil, currPoint);
            problem.AddResidualBlock(cost_function,NULL,parameter);

        }



        ceres::Solver::Options options;
        options.minimizer_type = ceres::TRUST_REGION;
//        options.initial_trust_region_radius = 1e1;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
//        options.dogleg_type = SUBSPACE_DOGLEG;
        options.max_num_iterations = 8;
        options.function_tolerance = 1e-5;
        options.max_num_consecutive_invalid_steps = 3;
        options.linear_solver_type = DENSE_QR;
        options.num_threads = 8;
        options.num_linear_solver_threads = 8;

        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        estimate1 = Eigen::Map< Eigen::Vector3d >(parameter);
        estimate = estimate1.cast<float>();


        Eigen::Affine2f transform(gridMapUtil.getTransformForState(estimate)); // transform: rotation->translation
        error=0;
        for (int i = 0; i < size; ++i) {

          const Eigen::Vector2f& currPoint (dataPoints.getVecEntry(i));

          Eigen::Vector3f transformedPointData(gridMapUtil.interpMapValueWithDerivatives(transform * currPoint));  /// {M,dM/dx,dM/dy}

          float funVal = 1.0f - transformedPointData[0];
    //      float weight=util::WeightValue(funVal);
          float weight=1.0;

          error+=funVal*weight;
        }

//        std::ofstream file("/home/wbw/error_original.txt",std::ofstream::app);
//        file<<std::setprecision(15)<<ros::Time::now().toSec()<<" "<<error<<"\n";
//        file.close();

//        std::ofstream ff("/home/wbw/report.txt",std::ofstream::app);
//        ff<<summary.FullReport()<<"\n\n";
//        ff.close();


    }

    void updateEstimatedPose(Eigen::Vector3f& estimate, const Eigen::Vector3f& change)
    {
        estimate += change;
    }

    void drawScan(const Eigen::Vector3f& pose, const ConcreteOccGridMapUtil& gridMapUtil, const DataContainer& dataContainer)
    {
        drawInterface->setScale(0.02);

        Eigen::Affine2f transform(gridMapUtil.getTransformForState(pose));

        int size = dataContainer.getSize();
        for (int i = 0; i < size; ++i) {
            const Eigen::Vector2f& currPoint (dataContainer.getVecEntry(i));
            drawInterface->drawPoint(gridMapUtil.getWorldCoordsPoint(transform * currPoint));
        }
    }

protected:
    Eigen::Vector3f dTr,dTr_last;
    Eigen::Matrix3f H,H_last;
    float error, error_last, lambda, lambda_max, lambda_min, err_original;
    Eigen::Vector3f estimate_last;
    int accept, begin_estimation;

    DrawInterface* drawInterface;
    HectorDebugInfoInterface* debugInterface;
};

}


#endif

