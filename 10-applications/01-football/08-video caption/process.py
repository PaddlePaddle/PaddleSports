class soccer_caption_generated():
    def __init__(self):
        self.action_event_list = ["Penalty", "Kick-off", "Goal", "Substitution", "Offside", "Shots on target", "Shots off target",
                                  "Clearance", "Ball out of play", "Throw-in", "Foul", "Indirect free-kick", "Direct free-kick",
                                  "Corner", "Yellow card", "Red card", "Yellow->red card"]
        self.action_event_list = [i.strip().lower() for i in self.action_event_list]
        self.action_event2id_dict =dict([[self.action_event_list[i],i] for i in range(len(self.action_event_list))])
        print(self.action_event2id_dict)
    def main(self,action_event_id,information):
        if isinstance(action_event_id,int):
            action_event_id = action_event_id
        else:
            action_event_id = self.action_event2id_dict[action_event_id.lower()]
        context_info_dict,content_info_dict = information
        context_cap_part = self.context_cap_gen(context_info_dict)
        # print(context_cap_part)
        if action_event_id==0:
            content_cap_part = self.penalty_cap_gen(content_info_dict)
        elif action_event_id==1:
            content_cap_part = self.kick_off_cap_gen(content_info_dict)
        elif action_event_id==2:
            content_cap_part = self.goal_cap_gen(content_info_dict)
        elif action_event_id==3:
            content_cap_part = self.substitution_cap_gen(content_info_dict)
        elif action_event_id==4:
            content_cap_part = self.offside_cap_gen(content_info_dict)
        elif action_event_id==5:
            content_cap_part = self.shots_on_target_cap_gen(content_info_dict)
        elif action_event_id==6:
            content_cap_part = self.shots_off_target_cap_gen(content_info_dict)
        elif action_event_id==7:
            content_cap_part = self.clearance_cap_gen(content_info_dict)
        elif action_event_id==8:
            content_cap_part = self.ball_out_of_play_cap_gen(content_info_dict)
        elif action_event_id==9:
            content_cap_part = self.throw_in_cap_gen(content_info_dict)
        elif action_event_id==10:
            content_cap_part = self.foul_cap_gen(content_info_dict)
        elif action_event_id==11:
            content_cap_part = self.indirect_free_kick_cap_gen(content_info_dict)
        elif action_event_id==12:
            content_cap_part = self.direct_free_kick_cap_gen(content_info_dict)
        elif action_event_id==13:
            content_cap_part = self.corner_cap_gen(content_info_dict)
        elif action_event_id==14:
            content_cap_part = self.yellow_card_cap_gen(content_info_dict)
        elif action_event_id==15:
            content_cap_part = self.red_card_cap_gen(content_info_dict)
        elif action_event_id==16:
            content_cap_part = self.yellow2red_card_cap_gen(content_info_dict)
        return context_cap_part+","+content_cap_part
    def penalty_cap_gen(self,content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team"}
        :return:A team win a penalty
        '''
        if content_info_dict.get("team_name"):
            gen_str = content_info_dict["team_name"]+" win a penalty."
            return gen_str
        else:
            raise ValueError("caption info about penalty event must have content_info_dict[\"team_name\"]")
        pass

    def kick_off_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team"}
        :return: A team takes a kick-off
        '''
        if content_info_dict.get("team_name"):
            gen_str = content_info_dict["team_name"] + " takes the kick-off."
            return gen_str
        else:
            raise ValueError("caption info about kick off event must have content_info_dict[\"team_name\"]")
        pass
    def goal_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":["A team","B team"],"team_score":[3,5],"goal_name":"A team"}
        :return: goal!A team3 B team5,A team footed shot.What a nice ball!
        '''
        gen_str = "goal!"
        if content_info_dict.get("team_name") and content_info_dict.get("team_score") and content_info_dict.get("goal_name"):
            gen_str0 = content_info_dict["team_name"][0]+str(content_info_dict["team_score"][0])+" "+content_info_dict["team_name"][1]+str(content_info_dict["team_score"][1])
            gen_str += gen_str0
            if isinstance(content_info_dict.get("goal_name"),list):
                gen_str0 =","+content_info_dict.get("goal_name")[0]+" "+content_info_dict.get("goal_name")[1]+" footed shot."
                gen_str += gen_str0
            elif isinstance(content_info_dict.get("goal_name"),str):
                gen_str0 =","+content_info_dict.get("goal_name")+" footed shot.What a nice ball!"
                gen_str += gen_str0

            return gen_str
        else:
            raise ValueError("caption info about goal event must have content_info_dict[\"team_name\"] [\"team_score\"] [\"goal_name\"]")
        pass
    def substitution_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":['player a','player b']}
        :return: substitution, A team, player a replaces player b.
        '''
        gen_str = "substitution, "
        if content_info_dict.get("team_name") and content_info_dict.get("player"):
            gen_str0 = content_info_dict["team_name"]+", "+content_info_dict["player"][0]+" replaces "+content_info_dict["player"][1]
            gen_str += gen_str0
            return gen_str+"."
        else:
            raise ValueError("caption info about Substitution event must have content_info_dict[\"team_name\"] [\"player\"]")
        pass
    def offside_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  A team player a is caught offside
        '''
        gen_str = "offside, "
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            gen_str+=" is caught offside. "
            return gen_str
        else:
            raise ValueError("caption info about offside event must have content_info_dict[\"team_name\"]or [\"player\"]")
        pass
    def shots_on_target_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  A team player a shoots on target.
        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            gen_str+=" shoots on target. "
            return gen_str
        else:
            raise ValueError("caption info about Shots_on_target event must have content_info_dict[\"team_name\"]or [\"player\"]")
        pass
    def shots_off_target_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  A team player a shoots off target.
        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            gen_str+=" shoots off target. "
            return gen_str
        else:
            raise ValueError("caption info about Shots_off_target event must have content_info_dict[\"team_name\"]or [\"player\"]")
        pass
    def clearance_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  A team player a clears the ball.

        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            gen_str+=" clears the ball. "
            return gen_str
        else:
            raise ValueError("caption info about clearance event must have content_info_dict[\"team_name\"]or [\"player\"]")
        pass
    def ball_out_of_play_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  the ball is out of play by player a

        '''
        gen_str = "the ball is out of play by "
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            gen_str+="."
            return gen_str
        else:
            raise ValueError("caption info about Ball out of play event must have content_info_dict[\"team_name\"]or [\"player\"]")
        pass
    def throw_in_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  ,A team player a commits a throw-in

        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            gen_str+=" commits a throw-in."
            return gen_str
        else:
            raise ValueError("caption info about throw-in event must have content_info_dict[\"team_name\"]or [\"player\"]")
        pass
    def foul_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  ,A team player a make a bad foul

        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            gen_str+=" commits a foul."
            return gen_str
        else:
            raise ValueError("caption info about make a bad foul event must have content_info_dict[\"team_name\"]or [\"player\"]")
    def indirect_free_kick_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  ,A team player wins an indirect free kick.

        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            gen_str+=" wins an indirect free kick. "
            return gen_str
        else:
            raise ValueError("caption info about Indirect_free_kick  event must have content_info_dict[\"team_name\"]or [\"player\"]")
    def direct_free_kick_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  ,A team player wins a direct free kick.

        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            gen_str+=" wins a direct free kick. "
            return gen_str
        else:
            raise ValueError("caption info about direct_free_kick  event must have content_info_dict[\"team_name\"]or [\"player\"]")

    def corner_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  ,corner,conceded by A team player a

        '''
        gen_str = "corner,conceded by "
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            return gen_str+"."
        else:
            raise ValueError("caption info about corner  event must have content_info_dict[\"team_name\"]or [\"player\"]")
    def yellow_card_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  A team player a is shown the yellow card for a foul

        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            return gen_str+" is shown the yellow card for a foul."
        else:
            raise ValueError("caption info about yellow card  event must have content_info_dict[\"team_name\"]or [\"player\"]")
    def red_card_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  A team player a is shown the red card for a bad foul.

        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            return gen_str+" is shown the red card for a bad foul."
        else:
            raise ValueError("caption info about red card  event must have content_info_dict[\"team_name\"]or [\"player\"]")
    def yellow2red_card_cap_gen(self, content_info_dict):
        '''

        :param content_info_dict: content_info_dict = {"team_name":"A team","player":'player a'}
        :return:  A team player a is shown the yellow card to red card for a bad foul

        '''
        gen_str = ""
        if content_info_dict.get("team_name") or content_info_dict.get("player"):
            gen_str0 = ""
            if content_info_dict.get("team_name"):
                gen_str0 = content_info_dict["team_name"]
            if content_info_dict.get("player"):
                gen_str0 +=" "+content_info_dict["player"]
            gen_str += gen_str0
            return gen_str+" is shown the yellow card to red card for a bad foul."
        else:
            raise ValueError("caption info about yellow2red card  event must have content_info_dict[\"team_name\"]or [\"player\"]")
    def context_cap_gen(self,context_info_dict):
        '''

        :param context_info: dict,key:[team_name,match_name,match_time]
        :return: during the Ateam VS Bteam Euro 2019 quarterfinal football match on March 20, 2019
        '''
        gen_str = "During the "
        if context_info_dict.get("team_name",None):
            if isinstance(context_info_dict["team_name"],list):
                context_info_dict["team_name"] = context_info_dict["team_name"][0]+" VS "+context_info_dict["team_name"][1]
            gen_str = gen_str + context_info_dict["team_name"]
        if context_info_dict.get("match_name"):
            gen_str = gen_str +" "+ context_info_dict["match_name"]
        gen_str = gen_str+" football match"
        if context_info_dict.get("match_time"):
            gen_str = gen_str +" on "+ context_info_dict["match_time"]
        return gen_str

# context_info_dict = {"team_name":["A team","B team"],"match_name":"Euro 2019 quarterfinal","match_time":"March 20 2019"}
context_info_dict = {"match_time":"March 20 2019"}
# context_info_dict = {}
content_info_dict = {"team_name":"A team"}#0 1
# content_info_dict = {}
# content_info_dict = {"team_name":["A team","B team"],"team_score":[3,5],"goal_name":"A team"} #2
# content_info_dict = {"team_name":"A team","player":["player a","player b"]} #3
content_info_dict = {"team_name":"A team","player":"player a"} #4 5 6 7 8 9 10 11 12 13
cap_gen = soccer_caption_generated()
a = cap_gen.main(10,[context_info_dict,content_info_dict]) #2代表事件索引
print(a)