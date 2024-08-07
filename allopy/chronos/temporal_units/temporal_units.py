from fractions import Fraction
from typing import Union

from allopy.topos.graphs import Tree
# from ..rhythm_trees import Meas, RhythmTree
from ..rhythm_trees import Meas, RhythmTree
from ..rhythm_trees.algorithms.subdivisions import measure_ratios, auto_subdiv
# from allopy.chronos.rhythm_trees.rt_algorithms import measure_ratios, auto_subdiv
from allopy.chronos.chronos import beat_duration, calc_onsets

# Prolationis Types
PULSTYPES = {'p', 'pulse', 'phase'}
DURTYPES  = {'d', 'duration', 'dur'}
RESTYPES  = {'r', 'rest', 'silence'}
ALLTYPES  = PULSTYPES | DURTYPES | RESTYPES

# Temporal Unit
class TemporalUnit:
    def __init__(self,
                 duration:int                    = 1,
                 tempus:Union[Meas,Fraction,str] = '1/1',
                 prolatio:Union[tuple,str]       = 'd',
                 tempo:Union[None,float]         = None,
                 beat:Union[str,Fraction]        = None):
        
        self.__type      = None
        self.__rtree     = self._set_rtree(duration, Meas(tempus), prolatio) # RhythmTree object
        self.__tempo     = tempo
        self.__beat      = self._set_beat(beat) # Fraction object
        self.__onsets    = None
        self.__durations = None
        self.__offset    = 0.0
    
    @classmethod
    def from_tree(cls, tree:Union[Tree, RhythmTree], tempo = None, beat = None):
        return cls(duration = tree.__duration if isinstance(tree, RhythmTree) else 1,
                   tempus   = tree._root,
                   prolatio = tree._children,
                   tempo    = tempo,
                   beat     = beat)

    @classmethod
    def from_subdivs(cls, subdivisions:tuple, duration:int = 1, tempo = None, beat = None):
        return cls(tempus   = Meas(sum(abs(r) * duration for r in measure_ratios(subdivisions))),
                   prolatio = subdivisions,
                   tempo    = tempo,
                   beat     = beat)

    @property
    def length(self):
        return self.__rtree.duration

    @property
    def tempus(self):
        return self.__rtree._root
    
    @property
    def prolationis(self):        
        return self.__rtree._children
    
    @property
    def ratios(self):
        return self.__rtree.ratios
    
    @property
    def rtree(self):
        return self.__rtree
    
    @property
    def tree(self):
        return self.__rtree.tree

    @property
    def tempo(self):
        return self.__tempo

    @property
    def beat(self):
        return self.__beat
    
    @property
    def type(self):
        return self.__type
    
    @property
    def offset(self):
        return self.__offset
    
    @property
    def onsets(self):
        if self.__tempo is None:
            return None
        if self.__onsets is None:
            self.__onsets = tuple(onset + self.__offset for onset in calc_onsets(self.durations))
        return self.__onsets

    @property
    def durations(self):
        if self.__tempo is None:
            raise ValueError('Tempo is not set')
        if self.__durations is None:
            self.__durations = tuple(
                beat_duration(ratio      = r,
                              bpm        = self.__tempo,
                              beat_ratio = self.__beat) for r in self.__rtree.ratios
                )
        return self.__durations

    @property
    def duration(self):
        if self.__tempo is None:
            raise ValueError('Tempo is not set')
        return sum(abs(d) for d in self.durations)
    
    @tempo.setter
    def tempo(self, tempo:Union[None,float,int]):
        self.__tempo     = tempo
        self.__onsets    = None
        self.__durations = None
    
    @beat.setter
    def beat(self, beat:Union[str,Fraction]):
        self.__beat      = Fraction(beat)
        self.__onsets    = None
        self.__durations = None
    
    @offset.setter
    def offset(self, offset:float):
        self.__offset = offset
        self.__onsets = None
        
    def decompose(self, prolatio:Union[RhythmTree,tuple,str] = 'd') -> 'UTSeq':
        if prolatio.lower() in {'s'}: prolatio = self.__rtree.subdivisions
        return UTSeq(TemporalUnit(tempus   = ratio,
                        prolatio = prolatio,
                        tempo    = self.__tempo,
                        beat     = self.__beat) for ratio in self.__rtree.ratios)

    def _set_rtree(self, duration:int, tempus:Meas, prolatio:Union[tuple,str]) -> RhythmTree:
        if isinstance(prolatio, tuple):
            r_tree = RhythmTree(duration = duration, meas = tempus, subdivisions = prolatio)
            self.__type = f'Ensemble ({r_tree.type})'
        elif isinstance(prolatio, str):
            prolatio = prolatio.lower()
            if prolatio in PULSTYPES:
                self.__type = 'Pulse'
                r_tree = RhythmTree(duration = duration, meas = tempus, subdivisions = (1,) * tempus.numerator)
            elif prolatio in DURTYPES:
                self.__type = 'Duration'
                r_tree = RhythmTree(duration = duration, meas = tempus, subdivisions = (1,))
            elif prolatio in RESTYPES:
                self.__type = 'Silence'
                r_tree = RhythmTree(duration = duration, meas = tempus, subdivisions = (-1,))
            else:
                raise ValueError(f'Invalid string: {prolatio}')
        else:
            raise ValueError(f'Invalid prolationis: {prolatio}')
        return r_tree
    
    def _set_beat(self, beat:Union[None,str,Fraction]) -> Fraction:
        if beat is None:
            return Fraction(1, self.__rtree.meas._denominator)
        return Fraction(beat)
    
    def __add__(self, other:Union['TemporalUnit', 'UTSeq', Fraction]):
        if isinstance(other, TemporalUnit):
            new_tempus = self.__rtree._root + other.__rtree._root
            return TemporalUnit(tempus   = new_tempus,
                      prolatio = 'd',
                      tempo    = self.__tempo,
                      beat     = self.__beat)
        elif isinstance(other, UTSeq):
            return UTSeq((self,) + other.uts)
        elif isinstance(other, Fraction):
            new_tempus = self.__rtree._root + other
            return TemporalUnit(tempus   = new_tempus,
                      prolatio = 'd',
                      tempo    = self.__tempo,
                      beat     = self.__beat)
        raise ValueError('Invalid Operand')

    def __sub__(self, other:Union['TemporalUnit', Fraction]):
        if isinstance(other, TemporalUnit):
            new_tempus = abs(self.__rtree._root - other.__rtree._root)
            return TemporalUnit(tempus   = new_tempus,
                      prolatio = 'd',
                      tempo    = self.__tempo,
                      beat     = self.__beat)
        elif isinstance(other, Fraction):
            new_tempus = abs(self.__rtree._root - other)
            return TemporalUnit(tempus   = new_tempus,
                      prolatio = 'd',
                      tempo    = self.__tempo,
                      beat     = self.__beat)
        raise ValueError('Invalid Operand')

    def __mul__(self, other:Union['TemporalUnit', Fraction, int]):
        if not isinstance(other, (TemporalUnit, Fraction, int)):
            raise ValueError('Invalid Operand')
        elif isinstance(other, TemporalUnit):
            new_tempus = self.__rtree._root * other.__rtree._root            
        else:
            new_tempus = self.__rtree._root * other
        return TemporalUnit(tempus   = new_tempus,
                  prolatio = self.__rtree.subdivisions,
                  tempo    = self.__tempo,
                  beat     = self.__beat)
    
    def __truediv__(self, other:Union['TemporalUnit', Fraction, int]):
        if isinstance(other, TemporalUnit):
            new_tempus = self.__rtree._root / other.__rtree._root
            return TemporalUnit(tempus   = new_tempus,
                      prolatio = self.__rtree.subdivisions,
                      tempo    = self.__tempo,
                      beat     = self.__beat)
        elif isinstance(other, (Fraction, int)):
            new_tempus = self.__rtree._root / other
            return TemporalUnit(tempus   = new_tempus,
                      prolatio = self.__rtree.subdivisions,
                      tempo    = self.__tempo,
                      beat     = self.__beat)
        raise ValueError('Invalid Operand')
    
    def __and__(self, other:'TemporalUnit'):
        if isinstance(other, TemporalUnit):
            return UTSeq((self, other))
        raise ValueError('Invalid Operand')
    
    def __iter__(self):
        return zip(self.onsets, self.durations)

    def __repr__(self):
        return (
            f'Type:        {self.__type}\n'
            f'Tempus:      {self.__rtree._root}\n'
            f'Tempo:       {self.__tempo}\n'
            f'Beat:        {self.__beat}\n'
            f'Prolationis: {self.__rtree.subdivisions}\n'
        )

# Temporal Unit Sequence
class UTSeq:
    def __init__(self, ut_seq:tuple[TemporalUnit]=(), offset:float=0.0):
        self.__seq    = ut_seq
        self.__offset = offset
        self.offset   = offset
        self._set_offsets()

    @property
    def uts(self):
        return self.__seq

    @property
    def onsets(self):
        return calc_onsets(self.durations)
    
    @property    
    def durations(self):
        return tuple(ut.duration for ut in self.__seq)
    
    @property
    def duration(self):
        return sum(self.durations)

    @property
    def T(self):
        return TB((UTSeq((ut,)) for ut in self.__seq))
    
    @property
    def offset(self):
        return self.__offset
    
    @offset.setter
    def offset(self, offset:float):
        self.__offset = offset
        for i, ut in enumerate(self.__seq):
            ut.offset = offset + sum(self.durations[j] for j in range(i))
            
    def _set_offsets(self):
        for i, ut in enumerate(self.__seq):
            ut.offset = self.__offset + sum(self.durations[j] for j in range(i))

    def __add__(self, other:Union[TemporalUnit, 'UTSeq']):
        if isinstance(other, TemporalUnit):
            return UTSeq(self.__seq + (other,))
        elif isinstance(other, UTSeq):
            return UTSeq(self.__seq + other.__seq)
        raise ValueError('Invalid Operand')

    def __and__(self, other:Union[TemporalUnit, 'UTSeq']):
        if isinstance(other, TemporalUnit):
            return TB((self.__seq, (other,)))
        elif isinstance(other, UTSeq):
            return TB((self.__seq, other.__seq))
        raise ValueError('Invalid Operand')

    def __iter__(self):
        # out = []
        # for i, ut in enumerate(self.__seq):
        #     for start, duration in ut:
        #         out.append((i, start + self.onsets[i], duration))
        # return iter(out)
        return zip(self.onsets, self.durations)

# Time Block
class TB:
    def __init__(self, tb:tuple[UTSeq]=(), offset:float=0.0, axis:float=0.0):
        self.__tb = tb
        self.__axis = axis
        self.__duration = max(ut_seq.duration for ut_seq in self.__tb) if self.__tb else 0.0

    @classmethod
    def from_tree_mat(cls, matrix, meas_denom:int=1, subdiv:bool=False,
                      rotation_offset:int=1, tempo=None, beat=None):
        tb = []
        for i, row in enumerate(matrix):
            seq = []
            for e in row:
                if subdiv:
                    D, S = e[0], auto_subdiv(e[1][::-1], i + rotation_offset)
                else:
                    D, S = e[0], e[1]
                seq.append(TemporalUnit(tempus   = Meas((D, meas_denom)),
                              prolatio = S,
                              tempo    = tempo,
                              beat     = beat))
            tb.append(UTSeq(seq))
        return cls(tuple(tb))

    @property
    def utseqs(self):
        return self.__tb

    @property
    def duration(self):
        return self.__duration

    @property
    def axis(self):
        return self.__axis

    # XXX - TO DO:
    @axis.setter
    def axis(self, axis):
        self.__axis = axis
        pass

    def tempo(self, tempo):
        for utseq in self.__tb:
            for ut in utseq:
                ut.tempo = tempo

    def beat(self, beat):
        for utseq in self.__tb:
            for ut in utseq:
                ut.beat = beat

    def __add__(self, other:Union[UTSeq, 'TB']):
        if isinstance(other, UTSeq):
            return TB(self.__tb + (other,))
        # elif isinstance(other, TB):
        #     return TB(self.__tb + other.__tb)
        raise ValueError('Invalid Operand')

    def __iter__(self):
        return iter(self.__tb)

# Temporal Block Sequence
class TBSeq:
    pass
    # def __init__(self, tb_seq:tuple[TB]=(), offset:float=0.0):
    #     self.__seq = tb_seq
    #     for i, tb in enumerate(self.__seq):
    #         tb.offset = offset + sum(tb_seq[j].duration for j in range(i))

    # @property
    # def tbs(self):
    #     return self.__seq

    # @property
    # def onsets(self):
    #     return calc_onsets(self.durations)
    
    # @property    
    # def durations(self):
    #     return tuple(tb.duration for tb in self.__seq)
    
    # @property
    # def duration(self):
    #     return sum(self.durations)

    # def __add__(self, other:Union[TB, 'TBSeq']):
    #     if isinstance(other, TB):
    #         return TBSeq(self.__seq + (other,))
    #     elif isinstance(other, TBSeq):
    #         return TBSeq(self.__seq + other.__seq)
    #     raise ValueError('Invalid Operand')

    # def __and__(self, other:Union[TB, 'TBSeq']):
    #     if isinstance(other, TB):
    #         return TBSeq((self.__seq, (other,)))
    #     elif isinstance(other, TBSeq):
    #         return TBSeq((self.__seq, other.__seq))
    #     raise ValueError('Invalid Operand')

    # def __iter__(self):
    #     return zip(self.onsets, self.durations)
