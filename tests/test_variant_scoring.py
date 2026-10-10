import unittest
import numpy as np
from common.variant_scoring import parse_mutations,mutated_sequence,score_from_table,ranking_metrics

class VariantScoringTests(unittest.TestCase):
    def test_position_and_sequence_validation(self):
        changes=parse_mutations('A1C:D3E','ACD')
        self.assertEqual(mutated_sequence('ACD',changes),'CCE')
        for text in ['C1A','A0C','D4E','A1X','A1C:A1D','']:
            with self.assertRaises(ValueError):parse_mutations(text,'ACD')
    def test_additive_scores_and_identity(self):
        ids={'A':0,'C':1,'D':2,'E':3};table=np.array([[1,3,0,0],[0,0,4,7]],dtype=float)
        changes=parse_mutations('A1C:D3E','ACD')
        self.assertEqual(score_from_table(changes,[1,3],table,ids),5.)
        self.assertEqual(score_from_table(parse_mutations('A1A','ACD'),[1,3],table,ids),0.)
        with self.assertRaises(ValueError):score_from_table(changes,[1],table[:1],ids)
    def test_log_normalizer_cancels(self):
        ids={'A':0,'C':1};v=np.array([[1.,4.,3.]])
        changes=parse_mutations('A1C','A');logp=v-np.log(np.exp(v).sum(-1,keepdims=True))
        self.assertAlmostEqual(score_from_table(changes,[1],v,ids),score_from_table(changes,[1],logp,ids))
    def test_rank_and_auc_ties(self):
        m=ranking_metrics([0,1,2,3],[0,1,2,3],[0,0,1,1]);self.assertAlmostEqual(m['spearman'],1.);self.assertEqual(m['auc'],1.)
        m=ranking_metrics([0,1,2,3],[1,1,1,1],[0,0,1,1]);self.assertIsNone(m['spearman']);self.assertEqual(m['auc'],.5)
    def test_invalid_labels(self):
        with self.assertRaises(ValueError):ranking_metrics([0,1,float('nan')],[0,1,2],[0,1,1])
        with self.assertRaises(ValueError):ranking_metrics([0,1,2],[0,1,2],[0,1,2])
