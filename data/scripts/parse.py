import re

class Player:
    fideid   = "NULL"
    sex      = "NULL"
    name     = "NULL"
    country  = "NULL"
    title    = "NULL"
    k        = "NULL"
    birthday = "NULL"

    def toCsv(self):
        return "{fideid}, {sex}, {name}, {rating}, {country}, {title}, {k}, {birthday} \n".format(
                    fideid=self.fideid,
                    sex=self.sex,
                    name=self.name,
                    rating=self.rating,
                    country=self.country,
                    title=self.title,
                    k=self.k,
                    birthday=self.birthday)

ignore = ["<playerslist>", "</playerslist>"]
header = "fideid, sex, name, rating, country, title, k, birthday \n"

def parse():
    with open("srl.xml", 'r') as xml, open("srl.csv", 'a') as csv:
        csv.write(header)

        player = None

        for line in xml:
            line = line.rstrip()
            line = line.replace(',', "")

            if line in ignore:
                pass

            elif line == "<player>":
                player = Player()
            elif line == "</player>":
                csv.write(player.toCsv())
                player = None

            else:
                attr, val = re.split('<|>', line)[1:3]
                if val == "":
                    val = "NULL"
                setattr(player, attr, val)


if __name__ == "__main__":
    parse()

