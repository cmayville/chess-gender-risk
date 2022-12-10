import subprocess
import os
import json


def getGames(_id, year, gamefile):
    with open(gamefile, 'a') as _file:

        i = 0
        moregames = True
        first = True
        deleteflag = False

        while moregames:


            req = """curl -s -X POST old.chesstempo.com/requests/gameslist.php -d "startIndex={game}&results=1000&currentFen=rnbqkbnr%2Fpppppppp%2F8%2F8%2F8%2F8%2FPPPPPPPP%2FRNBQKBNR%20w%20KQkq%20-%200%201&sort=date&dir=desc&pieceColour=either&gameResult=any&yearMin={year}&yearMax={year}&player1ID={chesstempoid}&subsetMinRating=all" """.format(
                    year=year, chesstempoid=_id, game=i)

            
            data = subprocess.check_output(req, shell=True)
            data = json.loads(data)

            if data["result"]["games"] == []:
                if first:
                    print("no games in 2019 from ct player", _id)
                    deleteflag = True
                moregames = False
            else:
                if first:
                    _file.write("white, oprating, pgn\n")
                first = False
                print("got game", i, "from ct player", _id)
                game = data["result"]["games"][0]
                white = 1 if game["white_id"] == _id else 0
                oprating = game["eloblack"] if white else game["elowhite"]
                pgn = " ".join(game["moves_san"])
                _file.write("{white}, {oprating}, {pgn}\n".format(
                                  white=white,
                                  oprating=oprating,
                                  pgn=pgn))

            i += 1

    # triggers on no games in 2019
    if deleteflag:
        os.remove(gamefile)


def getCTid(fideid):

    req = """curl -s -X POST old.chesstempo.com/requests/playerslist.php -d "startIndex=0&results=1&sort=surname&dir=asc&player1Fide={fideid}&numGames=1&gender=either" """.format(
            fideid=fideid)
    data = subprocess.check_output(req, shell=True)
    data = json.loads(data)

    if data["result"]["players"] == []:
        print("no player", fideid)
        return None
    else:
        print("got player", fideid)
        return data["result"]["players"][0]["player_id"]

def getAllGames(start, end):
    with open("srl.csv", 'r') as srl:
        first = True
        for i, line in enumerate(srl):
            if not first and (i >= start and i < end):
                fide_id, _ = line.split(",", 1)
                chesstempo_id = getCTid(fide_id)
                if chesstempo_id:
                    getGames(chesstempo_id, 2019, "games/" + str(fide_id) + ".csv")
            first = False
            
import threading

if __name__ == "__main__":
    start = 14452
    inc = 20000

    counter = start

    threads = []
    for i in range(20):
        thread = threading.Thread(target=getAllGames, args=(counter, counter+inc))
        threads.append(thread)
        thread.start()
        counter += inc

    for thread in threads:
        thread.join()

    print("done :)")





    
    
