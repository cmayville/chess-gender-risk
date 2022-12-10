import fileinput
import os
from enum import Enum
import re

import chess
import chess.engine


class Side(Enum):
    WHITE = 0
    BLACK = 1

def riskScore(cur_wdl, prev_wdl):
    loss1 = prev_wdl[0] - cur_wdl[0]
    loss2 = prev_wdl[2] - cur_wdl[2]
    absloss1, absloss2 = abs(loss1), abs(loss2)

    score = loss1 if absloss1 == min(absloss1, absloss2) else loss2
    return score

def analyze(moves, side, movetime_sec, engine):
    risk_scores = []

    board = chess.Board()
    limit = chess.engine.Limit(time=movetime_sec)

    # initial wdl = 194, 647, 159 by lc0
    prev_wdl = (194, 647, 159)
    
    current_side = Side.WHITE
    for move in moves:
        board.push_san(move)

        if current_side == side and '#' not in move and move != moves[-1]:
            with engine.analysis(board, limit=limit) as analysis:
                info = analysis.get()
                eng_score = info.get("score")

                if eng_score:
                    wdl = eng_score.wdl()
                    wdl = (wdl[0], wdl[1], wdl[2])
                    risk_scores.append(riskScore(wdl, prev_wdl))

        current_side = Side.BLACK if current_side == Side.WHITE else Side.WHITE


    return sum(risk_scores) / len(risk_scores)

def parse(line):
    """
    Returns
    -------
    [str] -- moves
    Side
    int -- oprating
    """
    extra = ["e.p.", "+"]

    
    line = line.split(',', 2)
    side = Side.WHITE if line[0] else Side.BLACK
    oprating = int(line[1].strip()) if line[1].strip() != "None" else None

    for rm in extra:
        line[2] = line[2].replace(rm, '')
    pgn = re.split('[0-9]+\.', line[2].strip())

    # trim off empty start
    if pgn[0] == '':
        pgn.pop(0)
 

    moves = []
    for move in pgn:
        move = move.split(' ', 1)
        moves.append(move[0].strip())
        if len(move) > 1:
            moves.append(move[1].strip())

    return (moves, side, oprating)

def process(filename):

    risks = []
    opscores = []

    os.chdir('/home/charli/school/ml/final/lc0/build/release')
    engine = chess.engine.SimpleEngine.popen_uci("./lc0")

    os.chdir('/home/charli/school/ml/final/')
    with open(filename, "r") as _file:

        first = True
        

        for line in _file:
            if not first:
                moves, side, oprating = parse(line)
                av_risk = analyze(moves, side, .05, engine)

                risks.append(av_risk)
                if oprating:
                    opscores.append(oprating)
            first = False
        
    engine.quit()

    

    player_risk = sum(risks) / len(risks)
    player_opsc = sum(opscores) / len(opscores)

    return (player_risk, player_opsc, len(opscores)) 
    #return (risks, player_opsc, len(risks))

class suppressSTDERR:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save = os.dup(2)
        os.dup2(self.null_fd, 2)

    def __exit__(self, type, value, traceback):
        os.dup2(self.save, 2)
        os.close(self.null_fd)

if __name__ == "__main__":
    with suppressSTDERR():
        already_processed = []
        with open("risks.csv", 'r') as _file:
            for line in _file:
                fideid = line.split(',')[0].strip()
                if fideid != "fideid":
                    already_processed.append(fideid)
            
        with open("risks.csv", 'a') as _file:
            #_file.write("fideid, riskscore, gamecount, oprating \n")
            
            i = 0
            for line in fileinput.input():
                i += 1
                prog = str(round(i / 4016, 2) * 100) + "%"

                line = line.strip()
                fideid = line.split('.')[0]
                if fideid not in already_processed:
                    riskscore, oprating, gamecount = process("selected_games/20/" + line)
                    _file.write("{fideid}, {riskscore}, {gamecount}, {oprating} \n".format(
                        fideid=fideid,
                        riskscore=riskscore,
                        gamecount=gamecount,
                        oprating=oprating))
                    print(prog, fideid, "done")
                else:
                    print(fideid, "processed earlier, skipping")
            
"""

if __name__ == "__main__":
    with suppressSTDERR():
        with open("var_risks.csv", 'a') as _file:
            _file.write("fideid, riskscores\n")
            
            for line in fileinput.input():
                line = line.strip()
                fideid = line.split('.')[0]
                risks, oprating, gamecount = process("games/" + line)
                riskscore = ", ".join(list(map(str, risks)))
                _file.write("{fideid}, {riskscore} \n".format(
                    fideid=fideid,
                    riskscore=riskscore))
                print(fideid, "done")
            
"""
