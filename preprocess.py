def preprocess(file_in, training_out, testing_out):
    " Preprocess the original letter-recognition data: Split up into training and testing data "
    fp1 = open(training_out, 'w')
    fp2 = open(testing_out, 'w')

    # Split after m samples
    m = 16000

    i = 1
    for line in open(file_in):

        l = line.strip().split(',')

        # Change last and first columns
        t = l[-1]
        l[-1] = l[0]
        l[0] = t

        if i <= m:
            fp1.write(" ".join(l) + "\n")
        else:
            fp2.write(" ".join(l) + "\n")
        i += 1
    fp1.close()
    fp2.close()
    print "preprocessed data from %s to %s and %s" % (file_in, training_out, testing_out)

if __name__ == "__main__":
    preprocess("data/letter-recognition.data", "data/letter-recognition-training.data", "data/letter-recognition-validation.data")