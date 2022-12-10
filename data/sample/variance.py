

def variance(data):
    n = len(data)
    mean = sum(data) / n

    deviations = [(x - mean) ** 2 for x in data]
    variance = sum(deviations) / n
    return variance

def stepVar(scores):
    upto = []
    variances = []
    for score in scores:
        upto.append(score)
        variances.append(variance(upto))

    return variances

if __name__ == "__main__":

    with open("var_risks.csv", 'r') as riskf, open("prog_var.csv", 'a') as varf:
        first = True
        varf.write("fideid, variances\n")

        for line in riskf:
            if not first:
                line = line.split(",")
                fideid = line[0].strip()

                scores = [float(s.strip()) for s in line[1:]]
                varns = stepVar(scores)
                varns = ", ".join([str(v) for v in varns])

                varf.write("{fideid}, {varns}\n".format(
                    fideid=fideid, varns=varns))

            first = False

    
    

        
