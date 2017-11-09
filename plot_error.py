import numpy as np
from matplotlib import pyplot as plt
import os

def main():
    result_name = "results2_v3.txt"
    result_smooth_name = "results_smooth.txt"

    max_angle = 90.0
    threshold_error = 1
    dAngle = (2*max_angle)/(2*max_angle)

    threshold_error_rate = 5
    max_threshold_error_rate = 30
    threshold_error_rate_step = 1

    threshold_error_rate_L2 = 10
    max_threshold_error_rate_L2 = 30
    threshold_error_rate_step_L2 = 1

    with open(result_name, "r") as fp:
        result = np.loadtxt(fp)

    try:
        with open(result_smooth_name, "r") as fp:
            result_smooth = np.loadtxt(fp)
    except Exception as ex:
        result_smooth = None

    sad = np.array([0.0, 0.0, 0.0])                             # sum of absolute difference
    adv = np.zeros((int(len(result)/2.0), 4))                   # histogram of absolute error for each frame
    adh = np.zeros((int((2*max_angle)/dAngle), 3))              # histogram of absolute error for each angle
    adhn = np.zeros((int((2*max_angle)/dAngle), 3))             # support structure use for normalized adh
    aeh = np.zeros((int((2*max_angle)/threshold_error), 3))     # histogram of absolute error
    error_rate = np.zeros((int((max_threshold_error_rate-threshold_error_rate)/threshold_error_rate_step), 3))     # Error rate
    error_rate_L2 = np.zeros((int((max_threshold_error_rate_L2-threshold_error_rate_L2)/threshold_error_rate_step_L2), 1))     # Error rate

    n_samples = 0
    for gt, pred in zip(result[::2], result[1::2]):
        agt = np.array([gt[1], gt[2], gt[3]])
        apred = np.array([pred[1], pred[2], pred[3]])

        ad = np.array([abs(agt[0]-apred[0]),
                       abs(agt[1]-apred[1]),
                       abs(agt[2]-apred[2])])

        sad += ad

        adv[n_samples, :] = np.array([n_samples, ad[0], ad[1], ad[2]])

        for i in xrange(3):
            adh[int((agt[i] + max_angle) / dAngle), i] += ad[i]
            adhn[int((agt[i] + max_angle) / dAngle), i] += 1

            aeh[int((ad[i])/threshold_error), i] += 1

        for t in xrange(threshold_error_rate, max_threshold_error_rate, threshold_error_rate_step):
            for i in xrange(3):
                if ad[i] < t:
                    error_rate[int((t-threshold_error_rate)/threshold_error_rate_step), i] += 1

        for t in xrange(threshold_error_rate_L2, max_threshold_error_rate_L2, threshold_error_rate_step_L2):
            a_norm = np.linalg.norm(agt-apred)
            if a_norm < t:
                error_rate_L2[int((t-threshold_error_rate_L2)/threshold_error_rate_step_L2)] += 1

        n_samples += 1

    ########################################################################################################################
    # Mean Absolute Error + Standard Deviation
    ########################################################################################################################
    gt = result[::2,1:]
    pred = result[1::2,1:]

    abs_error = abs(gt-pred)[:]

    # mad = sad / n_samples
    mad = abs_error.mean(axis=0)
    std_e = abs_error.std(axis=0)

    euc_d = np.linalg.norm(gt-pred, axis=1)
    med = euc_d.mean()
    std_ed = euc_d.std()	
    
    th_accuracy = 10.0
    accuracy = (euc_d[euc_d < th_accuracy]).shape[0] / float(euc_d.shape[0])

    ########################################################################################################################
    gt = result[::2]
    pred = result[1::2]

    diff_pred = (abs(gt[:,1:] - pred[:,1:]) == 0).astype(np.uint8)[:]

    s_rate = np.sum(np.product(diff_pred, axis=1))
    angle_rate = np.sum(diff_pred, axis=0)

    # print "Validation samples: {}".format(float(len(result)/2.0))
    print "Validation samples: {}".format(len(abs_error))
    print "SAD: {}".format(sad)
    print "Mean Absolute Error: {}".format(mad)
    print "STD: {}".format(std_e)
    print "Mean Euclidean Distance: {}".format(med)
    print "STD: {}".format(std_ed)
    print "Accuracy: {}".format(accuracy)
    print "SUCCESS RATE: {}, ({}/{})".format(s_rate / float(len(diff_pred)), s_rate, len(diff_pred))
    print "ANGLE1 RATE:  {}, ({}/{})".format(angle_rate[0] / float(len(diff_pred)), angle_rate[0], len(diff_pred))
    print "ANGLE2 RATE:  {}, ({}/{})".format(angle_rate[1] / float(len(diff_pred)), angle_rate[1], len(diff_pred))
    print "ANGLE3 RATE:  {}, ({}/{})".format(angle_rate[2] / float(len(diff_pred)), angle_rate[2], len(diff_pred))

    ########################################################################################################################
    # GT vs Prediction error
    ########################################################################################################################
    plt.figure("CNN error")
    plt.subplot(331)
    plt.title('Roll')
    plt.plot(gt[:,0], gt[:,1], 'k', pred[:,0], pred[:,1], 'r')
    plt.ylim([-90,90])

    plt.subplot(334)
    plt.title('Pitch')
    plt.plot(gt[:,0], gt[:,2], 'k', pred[:,0], pred[:,2], 'g')
    plt.ylim([-90,90])

    plt.subplot(337)
    plt.title('Yaw')
    plt.plot(gt[:,0], gt[:,3], 'k', pred[:,0], pred[:,3], 'b')
    plt.ylim([-90,90])

    ########################################################################################################################
    # Frame error
    ########################################################################################################################
    plt.subplot(332)
    plt.title('Frame error - Roll')
    plt.plot(adv[:,0], adv[:,1], 'r')

    plt.subplot(335)
    plt.title('Frame error - Pitch')
    plt.plot(adv[:,0], adv[:,2], 'g')

    plt.subplot(338)
    plt.title('Frame error - Yaw')
    plt.plot(adv[:,0], adv[:,3], 'b')

    ########################################################################################################################
    # Angle error
    ########################################################################################################################
    adh[adhn>0] /= adhn[adhn>0]
    adh /= np.max(adh, axis=0)
    max_angle = int(max_angle)
    dAngle = int(dAngle)

    plt.subplot(333)
    plt.title('Angle error - Roll')
    plt.bar(range(-max_angle, max_angle,dAngle), adh[:,0], color='r')
    plt.xlim(-max_angle,max_angle)

    plt.subplot(336)
    plt.title('Angle error - Pitch')
    plt.bar(range(-max_angle, max_angle,dAngle), adh[:,1], color='g')
    plt.xlim(-max_angle,max_angle)

    plt.subplot(339)
    plt.title('Angle error - Yaw')
    plt.bar(range(-max_angle, max_angle,dAngle), adh[:,2], color='b')
    plt.xlim(-max_angle,max_angle)

    ########################################################################################################################
    # Error distribution
    ########################################################################################################################
    plt.figure("Error distribution")

    plt.subplot(321)
    plt.title('Roll')
    plt.plot(gt[:,0], gt[:,1], 'k', pred[:,0], pred[:,1], 'r')
    plt.ylim([-90,90])

    plt.subplot(323)
    plt.title('Pitch')
    plt.plot(gt[:,0], gt[:,2], 'k', pred[:,0], pred[:,2], 'g')
    plt.ylim([-90,90])

    plt.subplot(325)
    plt.title('Yaw')
    plt.plot(gt[:,0], gt[:,3], 'k', pred[:,0], pred[:,3], 'b')
    plt.ylim([-90,90])

    aeh = aeh / len(gt)

    plt.subplot(322)
    plt.title('Error distribution - Roll')
    plt.bar(range(0, 2*max_angle, threshold_error), aeh[:,0], color='r')
    plt.xlim(0, 2*max_angle)
    plt.axvline(10.0, color='k', linestyle='--')

    plt.subplot(324)
    plt.title('Error distribution - Pitch')
    plt.bar(range(0, 2*max_angle, threshold_error), aeh[:,1], color='g')
    plt.xlim(0, 2*max_angle)
    plt.axvline(10.0, color='k', linestyle='--')

    plt.subplot(326)
    plt.title('Error distribution - Yaw')
    plt.bar(range(0, 2*max_angle, threshold_error), aeh[:,2], color='b')
    plt.xlim(0, 2*max_angle)
    plt.axvline(10.0, color='k', linestyle='--')

    ########################################################################################################################
    # Error rate
    ########################################################################################################################
    plt.figure("Error rate")

    plt.subplot(331)
    plt.title('Roll')
    plt.plot(gt[:,0], gt[:,1], 'k', pred[:,0], pred[:,1], 'r')
    plt.ylim([-90,90])

    plt.subplot(334)
    plt.title('Pitch')
    plt.plot(gt[:,0], gt[:,2], 'k', pred[:,0], pred[:,2], 'g')
    plt.ylim([-90,90])

    plt.subplot(337)
    plt.title('Yaw')
    plt.plot(gt[:,0], gt[:,3], 'k', pred[:,0], pred[:,3], 'b')
    plt.ylim([-90,90])

    accuracy = error_rate / len(gt)
    accuracy_L2 = error_rate_L2 / len(gt)

    plt.subplot(332)
    plt.title('Accuracy - Roll')
    plt.plot(range(threshold_error_rate, max_threshold_error_rate, threshold_error_rate_step), accuracy[:,0], color='r')
    plt.ylim(0.5, 1.0)

    plt.subplot(335)
    plt.title('Accuracy - Pitch')
    plt.plot(range(threshold_error_rate, max_threshold_error_rate, threshold_error_rate_step), accuracy[:,1], color='g')
    plt.ylim(0.5, 1.0)

    plt.subplot(338)
    plt.title('Accuracy - Yaw')
    plt.plot(range(threshold_error_rate, max_threshold_error_rate, threshold_error_rate_step), accuracy[:,2], color='b')
    plt.ylim(0.5, 1.0)

    plt.subplot(336)
    plt.title('Accuracy - L2')
    plt.plot(range(threshold_error_rate_L2, max_threshold_error_rate_L2, threshold_error_rate_step_L2), accuracy_L2[:], color='b')
    plt.ylim(0.5, 1.0)

    ########################################################################################################################
    # Comparison Smooth
    ########################################################################################################################
    if result_smooth is not None:
        pred_smooth = result_smooth[1::2]

        plt.figure("Smooth")

        plt.subplot(131)
        plt.title('Roll')
        plt.plot(gt[:,0], gt[:,1], 'k', pred[:,0], pred[:,1], 'r', pred_smooth[:,0], pred_smooth[:,1], 'y')

        plt.subplot(132)
        plt.title('Pitch')
        plt.plot(gt[:,0], gt[:,2], 'k', pred[:,0], pred[:,2], 'g', pred_smooth[:,0], pred_smooth[:,2], 'y')

        plt.subplot(133)
        plt.title('Yaw')
        plt.plot(gt[:,0], gt[:,3], 'k', pred[:,0], pred[:,3], 'b', pred_smooth[:,0], pred_smooth[:,3], 'y')

    ########################################################################################################################
    # SAVE GRAPHS
    ########################################################################################################################
    dir_graphs = "graphs"

    if not os.path.isdir(dir_graphs):
        os.mkdir(dir_graphs)

    gt[:,0] = range(gt.shape[0])
    pred[:,0] = range(gt.shape[0])
    gt_pred = np.hstack((gt, pred[:,1:]))

    aeh = np.hstack((np.array(range(1, 2*max_angle+1, threshold_error)).reshape((aeh.shape[0],1)), aeh))

    adh = np.hstack((np.array(range(-max_angle,max_angle,dAngle)).reshape((adh.shape[0],1)), adh))

    accuracy = np.hstack((np.array(range(threshold_error_rate,max_threshold_error_rate,threshold_error_rate_step)).reshape((accuracy.shape[0],1)), accuracy))
    accuracy_L2 = np.hstack((np.array(range(threshold_error_rate_L2,max_threshold_error_rate_L2,threshold_error_rate_step_L2)).reshape((accuracy_L2.shape[0],1)), accuracy_L2))

    np.savetxt(dir_graphs + "/pose_ground_truth.txt", gt, delimiter="\t")
    np.savetxt(dir_graphs + "/pose_cnn_pred.txt", pred, delimiter="\t")
    np.savetxt(dir_graphs + "/pose_gt_and_pred.txt", gt_pred, delimiter="\t")
    np.savetxt(dir_graphs + "/pose_frame_error.txt", adv, delimiter="\t")
    np.savetxt(dir_graphs + "/pose_error_distribution.txt", aeh, delimiter="\t")
    np.savetxt(dir_graphs + "/pose_angle_error.txt", adh, delimiter="\t")
    np.savetxt(dir_graphs + "/pose_pred_accuracy.txt", accuracy, delimiter="\t")
    np.savetxt(dir_graphs + "/pose_pred_accuracy_L2.txt", accuracy_L2, delimiter="\t")

    ########################################################################################################################
    plt.show()


if __name__ == "__main__":
    main()

