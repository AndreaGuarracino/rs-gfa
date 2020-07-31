use bstr::{BString, ByteSlice};
use std::ops::Range;

use nom::{bytes::complete::*, IResult};

use crate::gfa::*;
use crate::optfields::*;

#[derive(Debug, Clone, PartialEq)]
pub struct GAF<T: OptFields> {
    pub seq_name: BString,
    pub seq_len: usize,
    pub seq_range: Range<usize>,
    pub strand: Orientation,
    pub path: GAFPath,
    pub path_len: usize,
    pub path_range: Range<usize>,
    pub residue_matches: usize,
    pub block_length: usize,
    pub quality: u8,
    pub optional: T,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GAFStep {
    SegId(Orientation, BString),
    StableIntv(Orientation, BString, Range<usize>),
}

impl GAFStep {
    fn parse_orient(bytes: &[u8]) -> IResult<&[u8], Orientation> {
        use nom::{branch::alt, combinator::map};
        use Orientation::*;

        let fwd = map(tag(">"), |_| Forward);
        let bwd = map(tag("<"), |_| Backward);
        alt((fwd, bwd))(bytes)
    }

    pub fn parse_step(i: &[u8]) -> IResult<&[u8], GAFStep> {
        use nom::{
            character::complete::digit1,
            combinator::{map, opt},
            sequence::{preceded, separated_pair},
        };

        let (i, orient) = Self::parse_orient(i)?;
        let (i, name) = is_not("<>: \t\r\n")(i)?;
        let name = name.into();

        let parse_digits = map(digit1, |bs| {
            let s = unsafe { std::str::from_utf8_unchecked(bs) };
            s.parse::<usize>().unwrap()
        });

        let parse_range = preceded(
            tag(":"),
            separated_pair(&parse_digits, tag("-"), &parse_digits),
        );

        let (i, range) = opt(parse_range)(i)?;
        if let Some((start, end)) = range {
            let range = start..end;
            Ok((i, GAFStep::StableIntv(orient, name, range)))
        } else {
            Ok((i, GAFStep::SegId(orient, name)))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GAFPath {
    StableId(BString),
    OrientIntv(Vec<GAFStep>),
}

impl GAFPath {
    pub fn parse_path(i: &[u8]) -> IResult<&[u8], GAFPath> {
        use nom::{
            combinator::{opt, verify},
            multi::many1,
        };
        let (i, path) = opt(many1(GAFStep::parse_step))(i)?;

        if let Some(path) = path {
            Ok((i, GAFPath::OrientIntv(path)))
        } else {
            let (i, stable_id) = verify(is_not("\t"), |bs: &[u8]| {
                bs.find_byteset("><").is_none()
            })(i)?;
            Ok((i, GAFPath::StableId(stable_id.into())))
        }
    }
}

#[derive(Debug, Clone)]
pub struct PAF<T: OptFields> {
    pub query_seq_name: BString,
    pub query_seq_len: usize,
    pub query_seq_range: Range<usize>,
    pub strand: Orientation,
    pub target_seq_name: BString,
    pub target_seq_len: usize,
    pub target_seq_range: Range<usize>,
    pub residue_matches: usize,
    pub block_length: usize,
    pub quality: u8,
    pub optional: T,
}

fn parse_next<I, T>(mut input: I) -> Option<T>
where
    I: Iterator,
    I::Item: AsRef<[u8]>,
    T: std::str::FromStr,
{
    let tmp = input.next()?;
    let bytes = tmp.as_ref();
    std::str::from_utf8(bytes).ok().and_then(|p| p.parse().ok())
}

fn parse_seq_fields<I>(mut input: I) -> Option<(BString, usize, Range<usize>)>
where
    I: Iterator,
    I::Item: AsRef<[u8]>,
{
    let name = input.next()?.as_ref().into();
    let len = parse_next(&mut input)?;
    let start = parse_next(&mut input)?;
    let end = parse_next(&mut input)?;

    Some((name, len, start..end))
}

pub fn parse_paf<I, T>(mut input: I) -> Option<PAF<T>>
where
    I: Iterator,
    I::Item: AsRef<[u8]>,
    T: OptFields,
{
    let (query_seq_name, query_seq_len, query_seq_range) =
        parse_seq_fields(&mut input)?;

    let strand = input.next().and_then(Orientation::from_bytes)?;

    let (target_seq_name, target_seq_len, target_seq_range) =
        parse_seq_fields(&mut input)?;

    let residue_matches = parse_next(&mut input)?;
    let block_length = parse_next(&mut input)?;
    let quality = parse_next(&mut input)?;

    let optional = T::parse(input);

    Some(PAF {
        query_seq_name,
        query_seq_len,
        query_seq_range,
        strand,
        target_seq_name,
        target_seq_len,
        target_seq_range,
        residue_matches,
        block_length,
        quality,
        optional,
    })
}

// Since GAF and PAF are *essentially* the same, we just reuse the PAF
// parser and add a check that the path matches the spec regex
pub fn parse_gaf<I, T>(input: I) -> Option<GAF<T>>
where
    I: Iterator,
    I::Item: AsRef<[u8]>,
    T: OptFields,
{
    let paf: PAF<T> = parse_paf(input)?;
    let (_, path) = GAFPath::parse_path(&paf.target_seq_name).ok()?;

    Some(GAF {
        path,
        seq_name: paf.query_seq_name,
        seq_len: paf.query_seq_len,
        seq_range: paf.query_seq_range,
        strand: paf.strand,
        path_len: paf.target_seq_len,
        path_range: paf.target_seq_range,
        residue_matches: paf.residue_matches,
        block_length: paf.block_length,
        quality: paf.quality,
        optional: paf.optional,
    })
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CIGAROp {
    M = 0,
    I = 1,
    D = 2,
    N = 3,
    S = 4,
    H = 5,
    P = 6,
    E = 7,
    X = 8,
}

impl CIGAROp {
    fn to_u8(self) -> u8 {
        use CIGAROp::*;
        match self {
            M => b'M',
            I => b'I',
            D => b'D',
            N => b'N',
            S => b'S',
            H => b'H',
            P => b'P',
            E => b'=',
            X => b'X',
        }
    }

    fn from_u8(byte: u8) -> Option<CIGAROp> {
        use CIGAROp::*;
        match byte {
            b'M' => Some(M),
            b'I' => Some(I),
            b'D' => Some(D),
            b'N' => Some(N),
            b'S' => Some(S),
            b'H' => Some(H),
            b'P' => Some(P),
            b'=' => Some(E),
            b'X' => Some(X),
            _ => None,
        }
    }
}

impl std::fmt::Display for CIGAROp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sym = char::from(self.to_u8());
        write!(f, "{}", sym)
    }
}

impl std::str::FromStr for CIGAROp {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.as_bytes()
            .get(0)
            .cloned()
            .and_then(CIGAROp::from_u8)
            .ok_or("Could not parse CIGAR operation")
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CIGAR(pub Vec<(u32, CIGAROp)>);

impl CIGAR {
    fn parse_op_cmd(input: &[u8]) -> IResult<&[u8], CIGAROp> {
        use nom::{branch::alt, combinator::map};
        use CIGAROp::*;
        alt((
            map(tag("M"), |_| M),
            map(tag("I"), |_| I),
            map(tag("D"), |_| D),
            map(tag("N"), |_| N),
            map(tag("S"), |_| S),
            map(tag("H"), |_| H),
            map(tag("P"), |_| P),
            map(tag("="), |_| E),
            map(tag("X"), |_| X),
        ))(input)
    }

    pub fn parse(i: &[u8]) -> IResult<&[u8], Self> {
        use nom::{
            character::complete::digit1, combinator::map, multi::many1,
            sequence::pair,
        };
        map(
            many1(pair(
                map(digit1, |bs| {
                    let s = unsafe { std::str::from_utf8_unchecked(bs) };
                    s.parse::<u32>().unwrap()
                }),
                Self::parse_op_cmd,
            )),
            CIGAR,
        )(i)
    }

    /*
    fn parse_first(bytes: &[u8]) -> Option<((u32, CIGAROp), &[u8])> {
        if bytes[0].is_ascii_digit() {
            let op_ix = bytes.find_byteset(b"MIDNSHP=X")?;
            let num = std::str::from_utf8(&bytes[0..op_ix]).ok()?;
            let num: u32 = num.parse().ok()?;
            let op = CIGAROp::from_u8(bytes[op_ix])?;
            let rest = &bytes[op_ix + 1..];
            Some(((num, op), rest))
        } else {
            None
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return None;
        }
        let mut cigar: Vec<(u32, CIGAROp)> = Vec::new();
        let mut bytes = bytes;
        while bytes.len() > 0 {
            let (cg, rest) = Self::parse_first(bytes)?;
            cigar.push(cg);
            bytes = rest;
        }

        Some(CIGAR(cigar))
    }
    */
}

impl std::fmt::Display for CIGAR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (len, op) in self.0.iter() {
            write!(f, "{}{}", len, op)?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gaf_lines() {
        use GAFStep::*;
        use Orientation::*;

        type GAF = super::GAF<OptionalFields>;
        let gaf_in1 =
            b"read1\t6\t0\t6\t+\t>s2>s3>s4\t12\t2\t8\t6\t6\t60\tcg:Z:6M";

        let path_i1: Vec<GAFStep> = ["s2", "s3", "s4"]
            .iter()
            .map(|&s| SegId(Forward, s.into()))
            .collect();

        let expected_1 = GAF {
            seq_name: "read1".into(),
            seq_len: 6,
            seq_range: 0..6,
            strand: Forward,
            path: GAFPath::OrientIntv(path_i1),
            path_len: 12,
            path_range: 2..8,
            residue_matches: 6,
            block_length: 6,
            quality: 60,
            optional: vec![OptField::new(b"cg", OptFieldVal::Z("6M".into()))],
        };

        let gaf_1: Option<GAF> = parse_gaf(gaf_in1.split_str("\t"));

        assert_eq!(Some(expected_1.clone()), gaf_1);

        let gaf_in2 = b"read1\t6\t0\t6\t+\tchr1\t12\t2\t8\t6\t6\t60\tcg:Z:6M";

        let expected_2 = GAF {
            path: GAFPath::StableId("chr1".into()),
            ..expected_1
        };

        let gaf_2: Option<GAF> = parse_gaf(gaf_in2.split_str("\t"));

        assert_eq!(Some(expected_2.clone()), gaf_2);

        let gaf_in3 =
            b"read2\t7\t0\t7\t-\t>chr1:5-8>foo:8-16\t11\t1\t8\t7\t7\t60\tcg:Z:7M";

        let path_i3: Vec<GAFStep> = vec![
            StableIntv(Forward, "chr1".into(), 5..8),
            StableIntv(Forward, "foo".into(), 8..16),
        ];

        let expected_3 = GAF {
            seq_name: "read2".into(),
            seq_len: 7,
            seq_range: 0..7,
            strand: Backward,
            path: GAFPath::OrientIntv(path_i3),
            path_len: 11,
            path_range: 1..8,
            residue_matches: 7,
            block_length: 7,
            quality: 60,
            optional: vec![OptField::new(b"cg", OptFieldVal::Z("7M".into()))],
        };

        let gaf_3: Option<GAF> = parse_gaf(gaf_in3.split_str("\t"));

        assert_eq!(Some(expected_3), gaf_3);
    }

    #[test]
    fn parse_gaf_step() {
        use GAFStep::*;
        use Orientation::*;

        // segment ids
        let s1 = b">s1";
        let s2 = b"<segmentid>s1<s2";

        let (i1, step1) = GAFStep::parse_step(s1).unwrap();
        // The step is parsed as an oriented segment ID
        assert_eq!(SegId(Forward, "s1".into()), step1);
        // If there's just one segment ID to parse, it consumes the entire input
        assert_eq!(b"", i1);

        let (i2, step2) = GAFStep::parse_step(s2).unwrap();
        assert_eq!(SegId(Backward, "segmentid".into()), step2);
        assert_eq!(b">s1<s2", i2);

        // Can parse another step from the remaining bytes
        let (i2_2, step2_2) = GAFStep::parse_step(i2).unwrap();
        assert_eq!(b"<s2", i2_2);
        assert_eq!(SegId(Forward, "s1".into()), step2_2);

        // stable intervals
        let s3 = b">chr1:123-456";
        let s4 = b"<chr2:123-456<chr2:455-780";

        let (i3, step3) = GAFStep::parse_step(s3).unwrap();
        assert_eq!(b"", i3);
        assert_eq!(StableIntv(Forward, "chr1".into(), 123..456), step3);

        let (i4, step4) = GAFStep::parse_step(s4).unwrap();
        assert_eq!(b"<chr2:455-780", i4);
        assert_eq!(StableIntv(Backward, "chr2".into(), 123..456), step4);

        let (i4_2, step4_2) = GAFStep::parse_step(i4).unwrap();
        assert_eq!(b"", i4_2);
        assert_eq!(StableIntv(Backward, "chr2".into(), 455..780), step4_2);

        // Stops at tabs
        let with_tab = b"<s2\t266";
        let (i, s) = GAFStep::parse_step(with_tab).unwrap();
        assert_eq!(b"\t266", i);
        assert_eq!(SegId(Backward, "s2".into()), s);

        // Must start with > or <

        // If the ID is followed by a :, that must be followed by an interval
    }

    #[test]
    fn parse_gaf_paths() {
        use GAFPath::*;
        use GAFStep::*;
        use Orientation::*;

        use std::ops::Range;

        let seg_fwd = |bs: &str| SegId(Forward, bs.into());
        let seg_bwd = |bs: &str| SegId(Backward, bs.into());
        let stbl_fwd =
            |bs: &str, r: Range<usize>| StableIntv(Forward, bs.into(), r);
        let stbl_bwd =
            |bs: &str, r: Range<usize>| StableIntv(Backward, bs.into(), r);

        // stable IDs
        let p_id1 = b"some_id1";
        let p_id2 = b"chr1\t123";

        let (i, p) = GAFPath::parse_path(p_id1).unwrap();
        assert_eq!(b"", i);
        assert_eq!(StableId("some_id1".into()), p);

        let (i, p) = GAFPath::parse_path(p_id2).unwrap();
        assert_eq!(b"\t123", i);
        assert_eq!(StableId("chr1".into()), p);

        println!("{:?}", GAFPath::parse_path(p_id1));
        println!("{:?}", GAFPath::parse_path(p_id2));

        // oriented paths

        let p_orient1 = b">s1>s2<s3<s4";
        let p_orient2 = b">chr1:5-8>foo:8-16<bar:16-20\t298";

        let (i, p) = GAFPath::parse_path(p_orient1).unwrap();
        assert_eq!(b"", i);
        assert_eq!(
            OrientIntv(vec![
                seg_fwd("s1"),
                seg_fwd("s2"),
                seg_bwd("s3"),
                seg_bwd("s4")
            ]),
            p
        );

        let (i, p) = GAFPath::parse_path(p_orient2).unwrap();
        assert_eq!(b"\t298", i);
        assert_eq!(
            OrientIntv(vec![
                stbl_fwd("chr1", 5..8),
                stbl_fwd("foo", 8..16),
                stbl_bwd("bar", 16..20),
            ]),
            p
        );

        // If the path doesn't start with an orientation, it must be a
        // stable ID, and thus cannot contain any > or <
        let err_input = b"s1>s2<s3\t123";
        let parse_error: IResult<&[u8], GAFPath> =
            GAFPath::parse_path(err_input);
        assert!(parse_error.is_err());
    }

    #[test]
    fn cigar_display() {
        let input = b"20M12D3M4N9S10H5P11=9X";
        let input_str = std::str::from_utf8(input).unwrap();
        let cigar = CIGAR::parse(input).unwrap().1;
        let cigstr = cigar.to_string();
        assert_eq!(input_str, cigstr);
    }

    #[test]
    fn cigar_parser() {
        use CIGAROp::*;

        let input = b"20M12D3M4N9S10H5P11=9X";
        let (i, cigar) = CIGAR::parse(input).unwrap();
        assert_eq!(b"", i);
        assert_eq!(
            CIGAR(vec![
                (20, M),
                (12, D),
                (3, M),
                (4, N),
                (9, S),
                (10, H),
                (5, P),
                (11, E),
                (9, X)
            ]),
            cigar
        );

        let input = b"20M12D93  X";
        let (i, cigar) = CIGAR::parse(input).unwrap();
        assert_eq!(b"93  X", i);
        assert_eq!(CIGAR(vec![(20, M), (12, D)]), cigar);

        assert!(CIGAR::parse(b"M20").is_err());
        assert!(CIGAR::parse(b"20").is_err());
        assert!(CIGAR::parse(b"").is_err());
    }
}
