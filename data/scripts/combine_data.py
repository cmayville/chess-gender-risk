

def combine():
    players = {}
    with open("srl.csv", 'r') as srl_file:
        for line in srl_file:
            fideid = line.split(',')[0].strip()
            if fideid != "fideid":
                players[fideid] = line.strip()
            
            
    with open("risks.csv", 'r') as risk_file, open("player_risk_2019_20games.csv", 'a') as _file:
            _file.write("fideid, sex, name, rating, country, title, k, birthday, avrisk, gamecount, oprating \n")

            for line in risk_file:
                fideid = line.split(',')[0].strip()
                rest = ", ".join([l.strip() for l in line.split(',')[1:]])
                if fideid != "fideid":
                    _file.write(players[fideid] + ", " + rest + " \n")
                
            

if __name__ == "__main__":
    combine()
