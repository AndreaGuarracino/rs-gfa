use nom::branch::alt;
use nom::bytes::complete::*;
use nom::character::complete::*;
use nom::combinator::map;
use nom::error::ErrorKind;
use nom::multi::separated_list;
use nom::sequence::{preceded, terminated};
use nom::Err;
use nom::IResult;

// #[macro_use]
use nom::regex::Regex;

use crate::gfa::*;

fn segment_ex() -> String {
    format!("S\t11\tACCTT\tRC:i:123")
}

lazy_static! {
    static ref RE_ORIENT: Regex = Regex::new(r"+|-").unwrap();
    static ref RE_OVERLAP: Regex = Regex::new(r"\*|([0-9]+[MIDNSHPX=])+").unwrap();
}

fn parse_name(input: &str) -> IResult<&str, String> {
    let (i, name) = re_find!(input, r"^[!-)+-<>-~][!-~]*")?;
    Ok((i, name.to_string()))
}

fn parse_header(input: &str) -> IResult<&str, Header> {
    let col = tag(":");
    let (i, _line_type) = terminated(tag("H"), tag("\t"))(input)?;
    let (i, _opt_tag) = terminated(tag("VN"), &col)(i)?;
    let (i, _opt_type) = terminated(tag("Z"), &col)(i)?;
    let (i, version) = re_find!(i, r"[ !-~]+")?;

    Ok((
        i,
        Header {
            version: version.to_string(),
        },
    ))
}

fn parse_sequence(input: &str) -> IResult<&str, String> {
    let (i, seq) = re_find!(input, r"\*|[A-Za-z=.]+")?;
    Ok((i, seq.to_string()))
}

fn parse_orient(input: &str) -> IResult<&str, Orientation> {
    let fwd = map(tag("+"), |_| Orientation::Forward);
    let bkw = map(tag("-"), |_| Orientation::Backward);
    alt((fwd, bkw))(input)
}

fn parse_overlap(input: &str) -> IResult<&str, String> {
    let (i, overlap) = re_find!(input, r"\*|([0-9]+[MIDNSHPX=])+")?;
    Ok((i, overlap.to_string()))
}

/*
fn parse_optional(input: &str) -> IResult<&str, OptionalField> {
    let col = tag(":");
    let (i, opt_tag) = re_find!(input, r"^[A-Za-Z][A-Za-z0-9]")?;
    let (i, opt_type) = preceded(col, one_of("AifZJHB"))(i)?;

    let (i, opt_val) = match opt_type {
        'A' => ,
        'i' => true,
        'f' => true,
        'Z' => true,
        'J' => true,
        'H' => true,
        'B' => true,
    }

    // let (i, opt_typ) = terminated(one_of("AifZJHB"), col);
    // let (i, opt_tag) = re_find!(input, r"[A-Za-Z][A-Za-z0-9]")?;
    // let (
}
*/

fn parse_segment(input: &str) -> IResult<&str, Segment> {
    let tab = tag("\t");
    let (input, _line_type) = terminated(tag("S"), &tab)(input)?;

    let (input, name) = terminated(parse_name, &tab)(input)?;

    let (input, seq) = terminated(parse_sequence, &tab)(input)?;

    // TODO branch on the length of the remaining input to read the rest

    let result = Segment {
        name: name,
        sequence: seq,
        read_count: None,
        fragment_count: None,
        kmer_count: None,
        uri: None,
    };

    Ok((input, result))
}

fn parse_link(input: &str) -> IResult<&str, Link> {
    let tab = tag("\t");
    let (i, _line_type) = terminated(tag("L"), &tab)(input)?;

    let seg = terminated(parse_name, &tab);
    let orient = terminated(parse_orient, &tab);

    let (i, from_segment) = seg(i)?;
    let (i, from_orient) = orient(i)?;
    let (i, to_segment) = seg(i)?;
    let (i, to_orient) = orient(i)?;
    let (i, overlap) = terminated(parse_overlap, &tab)(i)?;

    let result = Link {
        from_segment,
        from_orient,
        to_segment,
        to_orient,
        overlap,
        map_quality: None,
        num_mismatches: None,
        read_count: None,
        fragment_count: None,
        kmer_count: None,
        edge_id: None,
    };

    Ok((i, result))
}

fn parse_containment(input: &str) -> IResult<&str, Containment> {
    let tab = tag("\t");
    let (i, _line_type) = terminated(tag("C"), &tab)(input)?;

    let seg = terminated(parse_name, &tab);
    let orient = terminated(parse_orient, &tab);

    let (i, container_name) = seg(i)?;
    let (i, container_orient) = orient(i)?;
    let (i, contained_name) = seg(i)?;
    let (i, contained_orient) = orient(i)?;
    let (i, pos) = terminated(digit0, &tab)(i)?;

    let (i, overlap) = terminated(parse_overlap, &tab)(i)?;

    let result = Containment {
        container_name,
        container_orient,
        contained_name,
        contained_orient,
        overlap,
        pos: pos.parse::<usize>().unwrap(),
        read_coverage: None,
        num_mismatches: None,
        edge_id: None,
    };

    Ok((i, result))
}

fn parse_path(input: &str) -> IResult<&str, Path> {
    let tab = tag("\t");

    let (i, _line_type) = terminated(tag("P"), &tab)(input)?;
    let (i, path_name) = terminated(parse_name, &tab)(i)?;
    let (i, segs) = terminated(parse_name, &tab)(i)?;
    let segment_names = segs.split_terminator(",").map(String::from).collect();
    let (i, overlaps) = separated_list(tag(","), parse_overlap)(i)?;

    let result = Path {
        path_name,
        segment_names,
        overlaps,
    };

    Ok((i, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_parse_header() {
        let hdr = "H\tVN:Z:1.0";
        let hdr_ = Header {
            version: "1.0".to_string(),
        };

        match parse_header(hdr) {
            Err(err) => {
                println!("{:?}", err);
                assert_eq!(true, false)
            }
            Ok((res, h)) => {
                println!("{:?}", h);
                assert_eq!(h, hdr_)
            }
        }
    }

    #[test]
    fn can_parse_segment() {
        let seg = "S	11	ACCTT	";
        let seg_ = Segment {
            name: "11".to_string(),
            sequence: "ACCTT".to_string(),
            read_count: None,
            fragment_count: None,
            kmer_count: None,
            uri: None,
        };
        match parse_segment(seg) {
            Err(err) => {
                println!("{:?}", err);
                assert_eq!(true, false)
            }
            Ok((res, s)) => {
                println!("{:?}", s);
                assert_eq!(s, seg_)
            }
        }
    }

    #[test]
    fn can_parse_link() {
        let link = "L	11	+	12	-	4M	";
        let link_ = Link {
            from_segment: "11".to_string(),
            from_orient: Orientation::Forward,
            to_segment: "12".to_string(),
            to_orient: Orientation::Backward,
            overlap: "4M".to_string(),
            map_quality: None,
            num_mismatches: None,
            read_count: None,
            fragment_count: None,
            kmer_count: None,
            edge_id: None,
        };
        match parse_link(link) {
            Err(err) => {
                println!("{:?}", err);
                assert_eq!(true, false)
            }
            Ok((res, l)) => {
                println!("{:?}", l);
                assert_eq!(l, link_)
            }
        }
    }

    #[test]
    fn can_parse_containment() {
        let cont = "C\t1\t-\t2\t+\t110\t100M	";

        let cont_ = Containment {
            container_name: "1".to_string(),
            container_orient: Orientation::Backward,
            contained_name: "2".to_string(),
            contained_orient: Orientation::Forward,
            overlap: "100M".to_string(),
            pos: 110,
            read_coverage: None,
            num_mismatches: None,
            edge_id: None,
        };

        match parse_containment(cont) {
            Err(err) => {
                println!("{:?}", err);
                assert_eq!(true, false)
            }
            Ok((res, c)) => {
                println!("{:?}", c);
                assert_eq!(c, cont_)
            }
        }
    }

    #[test]
    fn can_parse_path() {
        let path = "P\t14\t11+,12-,13+\t4M,5M";

        let path_ = Path {
            path_name: "14".to_string(),
            segment_names: vec!["11+".to_string(), "12-".to_string(), "13+".to_string()],
            overlaps: vec!["4M".to_string(), "5M".to_string()],
        };

        match parse_path(path) {
            Err(err) => {
                println!("{:?}", err);
                assert_eq!(true, false)
            }
            Ok((res, p)) => {
                println!("{:?}", p);
                assert_eq!(p, path_)
            }
        }
    }
}
