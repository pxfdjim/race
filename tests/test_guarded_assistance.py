import unittest
from dataclasses import replace

from src.guarded_assistance import GuardConfig,guard
from test_b_assisted import observation


def proposal(oid):
    return {'object_id':oid,'event_id':'','basis':'proposal','cost':.1}


class GuardedAssistanceTests(unittest.TestCase):
    def test_weak_B_proposal_cannot_move_an_existing_A_identity(self):
        seed=observation(1,1.);later=observation(2,2.)
        base={o.key:{'object_id':'O00001'} for o in (seed,later)}
        proposed={seed.key:proposal('OB00001'),later.key:proposal('OB00002')}
        links,stats=guard([seed,later],base,proposed,{seed.key},GuardConfig())
        self.assertEqual(links[later.key]['object_id'],'OB00001')
        self.assertTrue(stats['baseline_assigned_sequences_preserved'])

    def test_unmapped_A_identity_is_retained_despite_B_ambiguity(self):
        o=observation(1)
        links,_=guard([o],{o.key:{'object_id':'O00003'}},{o.key:proposal('OB00002')},set(),GuardConfig())
        self.assertEqual(links[o.key]['object_id'],'OA00003')

    def test_unknown_fragment_needs_nearby_compatible_waveform(self):
        seed=observation(1,1.);good=observation(2,2.);bad=observation(3,2.)
        bad.sequence=replace(bad.sequence,rf=bad.sequence.rf+30.)
        obs=[seed,good,bad];base={o.key:{'object_id':''} for o in obs};ps={o.key:proposal('OB00001') for o in obs}
        links,_=guard(obs,base,ps,{seed.key},GuardConfig())
        self.assertTrue(links[good.key]['object_id'])
        self.assertFalse(links[bad.key]['object_id'])

    def test_refilled_fragment_does_not_recursively_anchor_the_next(self):
        obs=[observation(1,1.),observation(2,4.),observation(3,7.)]
        base={o.key:{'object_id':''} for o in obs};ps={o.key:proposal('OB00001') for o in obs}
        links,_=guard(obs,base,ps,{obs[0].key},GuardConfig())
        self.assertTrue(links[obs[1].key]['object_id'])
        self.assertFalse(links[obs[2].key]['object_id'])

    def test_same_RF_PW_with_incompatible_PRI_or_model_is_not_refilled(self):
        seed=observation(1,1.);pri=observation(2,2.);model=observation(3,2.)
        pri.sequence=replace(pri.sequence,pri=500.)
        model.sequence=replace(model.sequence,model='jittered')
        obs=[seed,pri,model];base={o.key:{'object_id':''} for o in obs};ps={o.key:proposal('OB00001') for o in obs}
        links,_=guard(obs,base,ps,{seed.key},GuardConfig())
        self.assertTrue(all(not links[o.key]['object_id'] for o in (pri,model)))

    def test_source_identity_scope_is_respected(self):
        seed=observation(1,1.);other=observation(2,2.)
        other.sequence=replace(other.sequence,scope=('other','test','p1'))
        obs=[seed,other];base={o.key:{'object_id':''} for o in obs};ps={o.key:proposal('OB00001') for o in obs}
        links,_=guard(obs,base,ps,{seed.key},GuardConfig())
        self.assertFalse(links[other.key]['object_id'])


if __name__=='__main__':unittest.main()
