import random
from os_ken.base.app_manager import OSKenApp
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.dpid import dpid_to_str

class SDNQLTRController(OSKenApp):

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SDNQLTRController, self).__init__(*args, **kwargs)
        self.q_values = {}  # Store Q-values for routing
        self.trust_values = {}  # Trust values for nodes
        self.learning_rate = 0.5
        self.discount_factor = 0.9
        print("SDNQLTRController initialized")

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.logger.info("Handshake taken place with {}".format(dpid_to_str(datapath.id)))
        self.__add_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        pkt = packet.Packet(msg.data)
        in_port = msg.match['in_port']

        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            return

        src = eth.src
        dst = eth.dst

        # Determine action based on trust values and Q-learning
        action = self.trust_based_decision(datapath, src, dst)

        actions = [parser.OFPActionOutput(action)]
        
        # Handle buffer_id correctly
        if msg.buffer_id != ofproto.OFP_NO_BUFFER:
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port, actions=actions)
        else:
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=in_port, actions=actions, data=msg.data)
        
        self.logger.info(f"Packet from {src} to {dst} forwarded via port {action}")
        datapath.send_msg(out)

        # Update Q-table and trust values based on actual transmission success
        # Here you might want to use a more sophisticated success metric based on real transmission feedback.
        self.update_q_table(src, dst, action, reward=1)  # For simplicity, using 1 as reward for now
        self.update_trust(src, success_rate=0.9)  # Placeholder trust update

    def trust_based_decision(self, datapath, src, dst):
        ofproto = datapath.ofproto
        if src not in self.q_values:
            self.q_values[src] = {}
        if dst not in self.q_values[src]:
            self.q_values[src][dst] = {ofproto.OFPP_FLOOD: random.random()}  # Initialize with FLOOD action

        # Only choose actions based on trust values
        best_action = max(self.q_values[src][dst], key=self.q_values[src][dst].get)
        return best_action if self.trust_values.get(src, 1.0) >= 0.5 else ofproto.OFPP_FLOOD  # Use flood if trust is low

    def update_q_table(self, src, dst, action, reward):
        if src not in self.q_values:
            self.q_values[src] = {}
        if dst not in self.q_values[src]:
            self.q_values[src][dst] = {}

        old_value = self.q_values[src][dst].get(action, 0)
        next_max = max(self.q_values.get(dst, {}).get(src, {0: 0}).values())
        self.q_values[src][dst][action] = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.logger.info(f"Updated Q-value for {src}->{dst} action {action}: {self.q_values[src][dst][action]}")

    def update_trust(self, node, success_rate):
        self.trust_values[node] = 0.9 * self.trust_values.get(node, 1.0) + 0.1 * success_rate
        self.logger.info(f"Updated trust value for node {node}: {self.trust_values[node]}")

    def __add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        self.logger.info("Flow-Mod written to {}".format(dpid_to_str(datapath.id)))
        datapath.send_msg(mod)
