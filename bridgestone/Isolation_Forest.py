__author__ = 'linh'

import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import matplotlib.font_manager
from sklearn.ensemble import IsolationForest
import timeit

def main():
    ftest_file = "Bridgestone raw data_20161101/タイヤ情報一覧_renamed.csv"
    ftrain_file = "data 2-8/Supervised data 2 (20170301).csv"
    IsolationForrest(ftest_file, ftrain_file)
    # plot()


def IsolationForrest(ftest_file, ftrain_file):
    f = open(ftest_file, "rt", encoding="utf-8")
    dataTest = csv.reader(f, delimiter=",")
    # f = open("36 categories/2･2-D･D_R195_Mix.csv", "rt", encoding="utf-8")
    ftrain = open(ftrain_file, "rt", encoding="utf-8")
    dataTrain = csv.reader(ftrain, delimiter=",")
    next(dataTrain)
    y1 = []
    x1 = []

    ######## read training data#######################
    for line in dataTrain:
        if line[67]!= "" and line[40] != "" and int(line[67])!= 0:
            y1.append((float(line[14])-float(line[40])))
            x1.append(int(line[67]))
    ftrain.close()
    x1, y1 = zip(*sorted(zip(x1, y1)))



    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)

    ##################divide test data into shop################
    shop = {}
    next(dataTest)
    for line in dataTest:
        if line[3]!= "" and line[66]!= '' and int(line[66])> 0:
            l = []
            if line[28] == "新品":
                if line[14] != '' and line[40] != '' and line[67] != '' and int(line[67]) > 0:

                    d = int(line[67])
                    a = float(line[14]) - float(line[40])
                    l = [int(line[67]), (float(line[14])- float(line[40])) ]

            elif line[28] == "RT1":
                if line[15] != '' and float(line[15]) != 0 and line[40] != '' and line[68] != '' and int(line[68]) > 0:
                    l = [int(line[68]), (float(line[15])- float(line[40])) ]
            elif line[28] == "RT2":
                if line[15] != '' and float(line[15]) != 0 and line[40] != '' and line[69] != ''and int(line[69]) > 0:
                    l = [int(line[69]), (float(line[15])- float(line[40])) ]
            if len(l)!= 0:
                #for number of vehicle
                #---------------------
                if line[3] in shop:
                    shop[line[3]].append(l)
                else:
                    shop[line[3]] = [l]



    ################# train process ############################
    rng = np.random.RandomState(42)
    x1 = np.log(x1)
    X = np.column_stack((x1, y1))

    start = timeit.default_timer()
    clfsvm = svm.OneClassSVM(nu=0.5, kernel='rbf', gamma='auto')
    clfsvm.fit(X)
    train_svm_rbf = timeit.default_timer()-start

    start = timeit.default_timer()
    clfsvm2 = svm.OneClassSVM(nu=0.5, kernel='poly', gamma='auto')
    clfsvm2.fit(X)
    train_svm_poly = timeit.default_timer()-start

    start = timeit.default_timer()
    clf = IsolationForest(max_samples=500, random_state=rng )
    clf.fit(X)
    train_IF = timeit.default_timer()-start
    print(len(shop))



    # test = np.array(shop["D44"])
    # pred = clf.predict(test)
    # print(pred)
    #

    ############## predict process#####################
    result = []
    test_IF = 0
    test_svm_poly = 0
    test_svm_rbf = 0

    for s in shop:
        score = 0
        test = np.array(shop[s])


        test[:,0] = np.log(test[:,0])

        start = timeit.default_timer()
        pred = clf.predict(test)
        test_IF += timeit.default_timer()-start
        outlier = pred[pred==-1].size

        start = timeit.default_timer()
        predsvm = clfsvm.predict(test)
        test_svm_rbf += timeit.default_timer()-start
        outliersvm = predsvm[predsvm==-1].size

        start = timeit.default_timer()
        predsvm2 = clfsvm2.predict(test)
        test_svm_poly += timeit.default_timer()-start
        outliersvm2 = predsvm2[predsvm2==-1].size


        ################### calculate algorithm accuracy ##################
        resistance = np.array(np.power(np.e,test[:,0])/(test[:,1]+np.e-12))
        # resistance = np.array(test[:,0]/test[:,1])
        for i in range(0, len(resistance)):
            if resistance[i] >27727:
                score += np.abs(test[i,0]-27727*test[i,1])/np.sqrt(1+27727**2)
            elif resistance[i] < 2151:
                score += np.abs(test[i,0]-2151*test[i,1])/np.sqrt(1+2151**2)
        resistance[resistance>27727] = -1
        resistance[resistance<2151] = -1
        resistance[resistance!=-1] = 1
        re = resistance[resistance==-1].size



        acc = resistance + pred
        tp = acc[acc==-2].size
        tn = acc[acc==2].size
        accsvm = resistance + predsvm
        tpsvm = accsvm[accsvm==-2].size
        tnsvm = accsvm[accsvm==2].size
        accsvm2 = resistance + predsvm2
        tpsvm2 = accsvm2[accsvm2==-2].size
        tnsvm2 = accsvm2[accsvm2 == 2].size

        acc = round((tp+tn)/len(pred),2)*100
        accsvm = round((tpsvm+tnsvm)/len(pred),2)*100
        accsvm2 = round((tpsvm2+tnsvm2)/len(pred),2)*100




        try:
            precision = round(tp/outlier, 2)*100
        except:
            precision = np.nan
        try:
            precisionsvm = round(tpsvm/outliersvm,2)*100
        except:
            precisionsvm = np.nan
        try:
            precisionsvm2 = round(tpsvm2/outliersvm2,2)*100
        except:
            precisionsvm2 = np.nan

        try:
            recall = round(tp/re,2)*100
            recallsvm = round(tpsvm/re,2)*100
            recallsvm2 = round(tpsvm2/re,2)*100
        except:
            recall = np.nan
            recallsvm = np.nan
            recallsvm2 = np.nan

        try:
            fscore = round(2*tp/(outlier+re),2)
            fscoresvm = round(2*tpsvm/(outliersvm+re), 2)
            fscoresvm2 = round(2*tpsvm2/(outliersvm2+re), 2)
        except:
            fscore = np.nan
            fscoresvm = np.nan
            fscoresvm2 = np.nan

        # print(score/len(pred), len(pred), outlier)

        result.append([s,len(pred), re,#score/len(pred),
                       outlier, round(outlier*100/len(pred),2),recall, precision, acc, fscore,
                       outliersvm, round(outliersvm*100/len(pred),2), recallsvm, precisionsvm,accsvm,fscoresvm,
                       outliersvm2, round(outliersvm2*100/len(pred),2),recallsvm2, precisionsvm2,accsvm2,fscoresvm2])

        # if s == "D14":
        #     outlier = test[predsvm==-1]
        #     normal = test[predsvm!=-1]
        #     b1 = plt.scatter(x1, y1, c='blue', s=40)
        #     b2 = plt.scatter(outlier[:,0], outlier[:,1], c='blueviolet', s=40)
        #     c = plt.scatter(normal[:,0], normal[:,1], c='gold', s=40)
        #     yl = np.linspace(0, 16, 10)
        #     xl1 = np.array(np.log(yl) + np.log(2151))
        #     xl2 = np.array(np.log(yl) + np.log(27727))
        #     # xl1 = np.array(2151*yl)
        #     # xl2 = np.array(27727*yl)
        #     c2, = plt.plot(xl1, yl, c='red')
        #     c3, = plt.plot(xl2, yl, c='red')
        #     plt.axis('tight')
        #     plt.xlim((0, np.log(500000)))
        #     # plt.xlim((0,500000))
        #     plt.ylim((0, 16))
        #     plt.xlabel("Distance")
        #     plt.ylabel("Erosion")
        #     plt.legend([ b1, b2, c, c2],
        #                ["Train data",
        #                 "shop anomalies", "shop normal data", "separation line between outlier and normal area"],
        #                loc="upper left",
        #                prop=matplotlib.font_manager.FontProperties(size=11))
        #     plt.title(
        #         "Shop %s ; errors novel regular: %d/%d ; "
        #         % (s, len(outlier), len(test)))
        #     plt.savefig("mail/SVMrbfD14", dpi = 220)
        #     plt.close()
        #
        #     outlier = test[predsvm2==-1]
        #     normal = test[predsvm2!=-1]
        #     b1 = plt.scatter(x1, y1, c='blue', s=40)
        #     b2 = plt.scatter(outlier[:,0], outlier[:,1], c='blueviolet', s=40)
        #     c = plt.scatter(normal[:,0], normal[:,1], c='gold', s=40)
        #     yl = np.linspace(0, 16, 10)
        #     xl1 = np.array(np .log(yl) + np.log(2151))
        #     xl2 = np.array(np.log(yl) + np.log(27727))
        #     # xl1 = np.array(2151*yl)
        #     # xl2 = np.array(27727*yl)
        #     c2, = plt.plot(xl1, yl, c='red')
        #     c3, = plt.plot(xl2, yl, c='red')
        #     plt.axis('tight')
        #     plt.xlim((0, np.log(500000)))
        #     # plt.xlim((0,500000))
        #     plt.ylim((0, 16))
        #     plt.xlabel("Distance")
        #     plt.ylabel("Erosion")
        #     plt.legend([ b1, b2, c, c2],
        #                ["Train data",
        #                 "shop anomalies", "shop normal data", "separation line between outlier and normal area"],
        #                loc="upper left",
        #                prop=matplotlib.font_manager.FontProperties(size=11))
        #     plt.title(
        #         "Shop %s ; errors novel regular: %d/%d ; "
        #         % (s, len(outlier), len(test)))
        #     plt.savefig("mail/SVMpolyD14", dpi = 220)
        #     plt.close()
        #
        #     outlier = test[pred==-1]
        #     normal = test[pred!=-1]
        #     b1 = plt.scatter(x1, y1, c='blue', s=40)
        #     b2 = plt.scatter(outlier[:,0], outlier[:,1], c='blueviolet', s=40)
        #     c = plt.scatter(normal[:,0], normal[:,1], c='gold', s=40)
        #     yl = np.linspace(0, 16, 10)
        #     xl1 = np.array(np.log(yl) + np.log(2151))
        #     xl2 = np.array(np.log(yl) + np.log(27727))
        #     # xl1 = np.array(2151*yl)
        #     # xl2 = np.array(27727*yl)
        #     c2, = plt.plot(xl1, yl, c='red')
        #     c3, = plt.plot(xl2, yl, c='red')
        #     plt.axis('tight')
        #     plt.xlim((0, np.log(500000)))
        #     # plt.xlim((0,500000))
        #     plt.ylim((0, 16))
        #     plt.xlabel("Distance")
        #     plt.ylabel("Erosion")
        #     plt.legend([ b1, b2, c, c2],
        #                ["Train data",
        #                 "shop anomalies", "shop normal data", "separation line between outlier and normal area"],
        #                loc="upper left",
        #                prop=matplotlib.font_manager.FontProperties(size=11))
        #     plt.title(
        #         "Shop %s ; errors novel regular: %d/%d ; "
        #         % (s, len(outlier), len(test)))
        #     plt.savefig("mail/IForestD14", dpi = 220)
        #     plt.close()

    ############# Draw IForest result for each shop ####################
    # result = []
    # for s in shop:
    #     test = np.array(shop[s])
    #     clf.fit(X)
    #     pred = clf.predict(test)
    #     outlier = test[pred==-1]
    #     normal = test[pred!=-1]
    #     b1 = plt.scatter(x1, y1, c='blue', s=40)
    #     b2 = plt.scatter(outlier[:,0], outlier[:,1], c='blueviolet', s=40)
    #     c = plt.scatter(normal[:,0], normal[:,1], c='gold', s=40)
    #     plt.axis('tight')
    #     # plt.xlim((0, 50000))
    #     # plt.ylim((0, 16))
    #     plt.legend([b1, b2, c],
    #                ["training observations",
    #                 "testing anomal observations", "testing normal observations"],
    #                loc="upper left",
    #                prop=matplotlib.font_manager.FontProperties(size=11))
    #     plt.xlabel(
    #         "Shop %s ; errors novel regular: %d/%d ; "
    #         % (s, len(outlier), len(test)))
    #     plt.savefig("IForest/"+s, dpi = 220)
    #     plt.close()
    #     result.append([s, len(outlier), len(pred), round(len(outlier)*100/len(pred),2)])



    ################ Write outlier result of all algorithms ##################
    fwrite = open("outliernew.csv", "wt", encoding="utf-8")
    datawrite = csv.writer(fwrite, delimiter=",")
    datawrite.writerow(["shop", "total amount","obvious outlier",
                        "IForest outlier predict", "IForest outlier percentage", "IForest recall", "IForest precision", "Iforest accuracy","IF F-score",
                        "SVM rbf outlier amount", "SVM rbf outlier percentage", "SVM rbf recall", "SVM rbf precision", "SVM rbf accuracy", "SVM rbf F-score",
                        "SVM poly outlier amount", "SVM poly outlier percentage", "SVM poly recall", "SVM poly precision", "SVM poly accuracy", "SVM poly F-score"])
    for line in result:
        datawrite.writerow(line)

    print(train_IF, train_svm_rbf, train_svm_poly)
    print(test_IF, test_svm_rbf, test_svm_poly)



def plot():
    f = open("Bridgestone raw data_20161101/タイヤ情報一覧_renamed.csv", "rt", encoding="utf-8")
    dataTest = csv.reader(f, delimiter=",")
    # f = open("36 categories/2･2-D･D_R195_Mix.csv", "rt", encoding="utf-8")
    ftrain = open("data 2-8/Supervised data 2 (20170301).csv", "rt", encoding="utf-8")
    dataTrain = csv.reader(ftrain, delimiter=",")
    next(dataTrain)
    y1 = []
    x1 = []
    for line in dataTrain:
        if line[67]!= "" and line[40] != "" and int(line[67])!= 0:
            y1.append((float(line[14])-float(line[40])))
            x1.append(int(line[67]))
    ftrain.close()
    x1, y1 = zip(*sorted(zip(x1, y1)))

    shop = {}
    vehicle = {}
    next(dataTest)
    for line in dataTest:
        if line[3]!= "" and line[66]!= '' and int(line[66])> 0:
            l = []
            if line[28] == "新品":
                if line[14] != '' and line[40] != '' and line[67] != '' and int(line[67]) > 0:

                    d = int(line[67])
                    a = float(line[14]) - float(line[40])
                    l = [int(line[67]), (float(line[14])- float(line[40])) ]

            elif line[28] == "RT1":
                if line[15] != '' and float(line[15]) != 0 and line[40] != '' and line[68] != '' and int(line[68]) > 0:
                    l = [int(line[68]), (float(line[15])- float(line[40])) ]
            elif line[28] == "RT2":
                if line[15] != '' and float(line[15]) != 0 and line[40] != '' and line[69] != ''and int(line[69]) > 0:
                    l = [int(line[69]), (float(line[15])- float(line[40])) ]
            if len(l)!= 0:
                if line[3] in shop:
                    shop[line[3]].append(l)
                    vehicle[line[3]].append(line[6])
                else:
                    shop[line[3]] = [l]
                    vehicle[line[3]] = [line[6]]
    rng = np.random.RandomState(42)
    x1 = np.log(x1)
    X = np.column_stack((x1, y1))

    start = timeit.default_timer()
    clf = IsolationForest(max_samples=500, random_state=rng )
    clf.fit(X)

    result = []
    for s in shop:
        test = np.array(shop[s])


        test[:,0] = np.log(test[:,0])

        pred = clf.predict(test)
        test = np.array(shop[s])
        outlier = test[pred==-1]
        n0_outlier = len(outlier)
        n0_vehicle = len(set(vehicle[s]))
        result.append([s, len(pred), round(n0_outlier*100/len(pred),2), n0_vehicle])

        if s == "D100" or s == "D223" or s=="D160":
            print(pred)
            normal = test[pred!= -1]
            b2 = plt.scatter(outlier[:,0], outlier[:,1], c='blueviolet', s=40)
            c = plt.scatter(normal[:,0], normal[:,1], c='gold', s=40)
            plt.axis('tight')
            plt.xlabel('Distance')
            plt.ylabel('Erosion')
            plt.xlim((0, 400000))
            plt.ylim((0, 16))
            plt.legend([b2, c],
                     [ "anomal", "normal data"],
                       loc="upper left",
                       prop=matplotlib.font_manager.FontProperties(size=11))
            plt.xlabel(
                "Shop %s ; errors novel regular: %d/%d ; "
                % (s, len(outlier), len(test)))
            plt.savefig(s, dpi = 220)
            plt.close()


    result = np.array(result)
    # fwrite = open("shopranking.csv", "wt", encoding="utf-8")
    # datawrite = csv.writer(fwrite, delimiter=",")
    # datawrite.writerow(["shop", "total amount","obvious outlier percentage", "n0_vehicle"])
    #
    # for line in result:
    #     datawrite.writerow(line)

    outlier = result[:,2].astype(float)
    print(outlier[outlier>75].size)
    for i in range(0, len(result)):
        if float(result[i,1]) < 250 and float(result[i,2]) < 23:
            plt.plot(result[i,1], result[i,2], 'go', color = '#a9bcf5', markersize=float(result[i, 3]))
        elif float(result[i,1]) >= 250 and float(result[i,2]) < 30:
            plt.plot(result[i,1], result[i,2], 'go', color = '#0000ff', markersize=float(result[i, 3]))
        elif float(result[i,1]) >= 250 and float(result[i,2]) >75:
            plt.plot(result[i,1], result[i,2], 'go', color = '#df013a', markersize=float(result[i, 3]))
        elif float(result[i,1]) < 250 and float(result[i,2]) > 75:
            plt.plot(result[i,1], result[i,2], 'go', color = '#f5a9bc', markersize=float(result[i, 3]))
        else:
            plt.plot(result[i,1], result[i,2], 'yo', color = '#d0a9f5', markersize=float(result[i, 3]))

    plt.xlim(0, 400)
    plt.ylim(0,100)
    # xmax = max(result[:,1])
    # print(xmax)
    # plotlim = plt.xlim(0, 100) + plt.ylim(0, 200)
    # ax.imshow([[1,1],[0,0]], cmap=plt.cm.Reds, interpolation='bicubic', extent=plotlim)
    for label, x, y in zip(result[:,0], result[:, 1], result[:, 2]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(0, 0),
            textcoords='offset points', ha='right', va='bottom', size=10,

        )
    plt.xlabel("Data Amount")
    plt.ylabel("Outlier Percentage")

    plt.savefig('Amount4.png', dpi = 220)
    plt.show()

if __name__ == '__main__':
    main()