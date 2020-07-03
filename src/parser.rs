use nom::branch::alt;
use nom::bytes::complete::*;
use nom::character::complete::*;
use nom::combinator::map;
use nom::sequence::terminated;
use nom::IResult;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, Lines};
use std::path::PathBuf;

use lazy_static::lazy_static;
use regex::Regex;

use crate::gfa::FieldValue;
use crate::gfa::*;

fn get_optional_tag(opt: &Option<OptionalField>) -> Option<&str> {
    opt.as_ref().map(|f| f.tag.as_str())
}

fn get_optional_field<'a>(opts: &'a [OptionalField], tag: &str) -> Option<&'a OptionalField> {
    opts.iter().find(|o| o.tag == tag)
}

fn parse_optional_tag(input: &str) -> Option<String> {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"[A-Za-z][A-Za-z0-9]").unwrap();
    }
    RE.find(input).map(|s| s.as_str().to_string())
}

fn parse_optional_char(input: &str) -> Option<char> {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"[!-~]").unwrap();
    }

    RE.find(input).and_then(|s| s.as_str().chars().nth(0))
}

fn parse_optional_int(input: &str) -> Option<i64> {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"[-+]?[0-9]+").unwrap();
    }

    RE.find(input).and_then(|s| s.as_str().parse().ok())
}

fn parse_optional_float(input: &str) -> Option<f32> {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?").unwrap();
    }

    RE.find(input).and_then(|s| s.as_str().parse().ok())
}

fn parse_optional_string(input: &str) -> Option<String> {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"[ !-~]+").unwrap();
    }
    RE.find(input).map(|s| s.as_str().to_string())
}

fn parse_optional_bytearray(input: &str) -> Option<Vec<u32>> {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"[0-9A-F]+").unwrap();
    }

    RE.find(input)
        .map(|s| s.as_str().chars().filter_map(|c| c.to_digit(16)).collect())
}

fn parse_optional_array<T: std::str::FromStr>(input: &str) -> Option<Vec<T>> {
    input
        .split_terminator(',')
        .map(|f| f.parse().ok())
        .collect()
}

fn parse_optional_field(input: &str) -> Option<OptionalField> {
    use OptionalFieldValue::*;

    lazy_static! {
        static ref RE: Regex = Regex::new(r"[AifZJHB]").unwrap();
    }

    let fields: Vec<_> = input.split_terminator(':').collect();

    let field_type = RE.find(fields[1]).map(|s| s.as_str())?;
    let field_tag = parse_optional_tag(fields[0])?;
    let field_value = match field_type {
        // char
        "A" => parse_optional_char(fields[2]).map(PrintableChar),
        // int
        "i" => parse_optional_int(fields[2]).map(SignedInt),
        // float
        "f" => parse_optional_float(fields[2]).map(Float),
        // string
        "Z" => parse_optional_string(fields[2]).map(PrintableString),
        // JSON string
        "J" => parse_optional_string(fields[2]).map(JSON),
        // bytearray
        "H" => parse_optional_bytearray(fields[2]).map(ByteArray),
        "B" => {
            if fields[2].starts_with('f') {
                parse_optional_array(&fields[2][1..]).map(FloatArray)
            } else {
                parse_optional_array(&fields[2][1..]).map(IntArray)
            }
        }
        _ => panic!(
            "Tried to parse optional field with unknown type '{}'",
            fields[0]
        ),
    }?;

    Some(OptionalField {
        tag: field_tag,
        content: field_value,
    })
}

fn parse_header(input: &str) -> IResult<&str, Header> {
    use OptionalFieldValue::PrintableString;

    match parse_optional_field(input).map(|o| o.content) {
        Some(PrintableString(v)) => Ok((input, Header { version: Some(v) })),
        _ => Ok((input, Header { version: None })),
    }
}

fn parse_orient(input: &str) -> IResult<&str, Orientation> {
    let fwd = map(tag("+"), |_| Orientation::Forward);
    let bkw = map(tag("-"), |_| Orientation::Backward);
    alt((fwd, bkw))(input)
}

fn parse_name(input: &str) -> Option<String> {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"[!-)+-<>-~][!-~]*").unwrap();
    }

    RE.find(input).map(|s| s.as_str().to_string())
}

fn parse_sequence(input: &str) -> Option<String> {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"\*|[A-Za-z=.]+").unwrap();
    }

    RE.find(input).map(|s| s.as_str().to_string())
}

fn parse_segment(input: &str) -> IResult<&str, Segment> {
    let fields: Vec<_> = input.split_terminator('\t').collect();

    let name = parse_name(fields[0]).unwrap();
    let sequence = parse_sequence(fields[1]).unwrap();

    let opt_fields: Vec<_> = fields[2..]
        .into_iter()
        .filter_map(|f| parse_optional_field(*f))
        .collect();

    let segment_length: Option<i64> = get_optional_field(&opt_fields, "LN")
        .map(|o| o.content)
        .and_then(|o| {
            let x: Option<i64> = FieldValue::unwrap(o);
            x
        });

    let segment_length = get_optional_field(&opt_fields, "LN")
        .map(|o| o.content)
        .and_then(|o| o.unwrap_int());
    let read_count = get_optional_field(&opt_fields, "RC")
        .map(|o| o.content)
        .and_then(|o| o.unwrap_int());
    let fragment_count = get_optional_field(&opt_fields, "FC")
        .map(|o| o.content)
        .and_then(|o| o.unwrap_int());
    let kmer_count = get_optional_field(&opt_fields, "KC")
        .map(|o| o.content)
        .and_then(|o| o.unwrap_int());
    let sha256 = get_optional_field(&opt_fields, "SH")
        .map(|o| o.content)
        .and_then(|o| o.unwrap_bytearray());
    let uri = get_optional_field(&opt_fields, "UR")
        .map(|o| o.content)
        .and_then(|o| o.unwrap_string());

    let result = Segment {
        name,
        sequence,
        segment_length,
        read_count,
        fragment_count,
        kmer_count,
        sha256,
        uri,
    };

    Ok((input, result))
}

fn parse_link(input: &str) -> IResult<&str, Link> {
    let fields: Vec<_> = input.split_terminator('\t').collect();

    let from_segment = fields[0].to_string();
    let (_, from_orient) = parse_orient(fields[1])?;
    let to_segment = fields[2].to_string();
    let (_, to_orient) = parse_orient(fields[3])?;
    let overlap = fields[4].to_string();

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

    Ok((input, result))
}

fn parse_containment(input: &str) -> IResult<&str, Containment> {
    let fields: Vec<_> = input.split_terminator('\t').collect();

    let container_name = fields[0].to_string();
    let (_, container_orient) = parse_orient(fields[1])?;
    let contained_name = fields[2].to_string();
    let (_, contained_orient) = parse_orient(fields[3])?;
    let pos = fields[4];

    let overlap = fields[5].to_string();

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

    Ok((input, result))
}

fn parse_path(input: &str) -> IResult<&str, Path> {
    let fields: Vec<_> = input.split_terminator('\t').collect();

    let path_name = fields[0].to_string();

    let segment_names = fields[1].split_terminator(',').collect();
    let overlaps = fields[2].split_terminator(',').map(String::from).collect();

    let result = Path::new(&path_name, segment_names, overlaps);

    Ok((input, result))
}

pub fn parse_line(line: &str) -> IResult<&str, Line> {
    let (i, line_type) = terminated(one_of("HSLCP#"), tab)(line)?;

    match line_type {
        'H' => {
            let (i, h) = parse_header(i)?;
            Ok((i, Line::Header(h)))
        }
        '#' => Ok((i, Line::Comment)),
        'S' => {
            let (i, s) = parse_segment(i)?;
            Ok((i, Line::Segment(s)))
        }
        'L' => {
            let (i, l) = parse_link(i)?;
            Ok((i, Line::Link(l)))
        }
        'C' => {
            let (i, c) = parse_containment(i)?;
            Ok((i, Line::Containment(c)))
        }
        'P' => {
            let (i, p) = parse_path(i)?;
            Ok((i, Line::Path(p)))
        }
        _ => Ok((i, Line::Comment)), // ignore unrecognized headers for now
    }
}

pub fn parse_gfa_stream<'a, B: BufRead>(
    input: &'a mut Lines<B>,
) -> impl Iterator<Item = Line> + 'a {
    input.map(|l| {
        let l = l.expect("Error parsing file");
        let r = parse_line(&l);
        if let Ok((_, parsed)) = r {
            parsed
        } else {
            panic!("Error parsing GFA lines")
        }
    })
}

pub fn parse_gfa(path: &PathBuf) -> Option<GFA> {
    let file = File::open(path).unwrap_or_else(|_| panic!("Error opening file {:?}", path));

    let reader = BufReader::new(file);
    let lines = reader.lines();

    let mut gfa = GFA::new();

    for line in lines {
        let l = line.expect("Error parsing file");
        let p = parse_line(&l);

        if let Ok((_, Line::Segment(s))) = p {
            gfa.segments.push(s);
        } else if let Ok((_, Line::Link(l))) = p {
            gfa.links.push(l);
        } else if let Ok((_, Line::Containment(c))) = p {
            gfa.containments.push(c);
        } else if let Ok((_, Line::Path(pt))) = p {
            gfa.paths.push(pt);
        }
    }

    Some(gfa)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn can_parse_header() {
        let hdr = "VN:Z:1.0";
        let hdr_ = Header {
            version: Some("1.0".to_string()),
        };

        match parse_header(hdr) {
            Err(err) => {
                panic!(format!("{:?}", err));
            }
            Ok((_res, h)) => assert_eq!(h, hdr_),
        }
    }

    #[test]
    fn can_parse_segment() {
        let seg = "11	ACCTT	";
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
                panic!(format!("{:?}", err));
            }
            Ok((_res, s)) => assert_eq!(s, seg_),
        }
    }

    #[test]
    fn can_parse_link() {
        let link = "11	+	12	-	4M";
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
                panic!(format!("{:?}", err));
            }
            Ok((_res, l)) => assert_eq!(l, link_),
        }
    }

    #[test]
    fn can_parse_containment() {
        let cont = "1\t-\t2\t+\t110\t100M";

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
                panic!(format!("{:?}", err));
            }
            Ok((_res, c)) => assert_eq!(c, cont_),
        }
    }

    #[test]
    fn can_parse_path() {
        let path = "14\t11+,12-,13+\t4M,5M";

        let path_ = Path {
            path_name: "14".to_string(),
            segment_names: vec![
                ("11".to_string(), Orientation::Forward),
                ("12".to_string(), Orientation::Backward),
                ("13".to_string(), Orientation::Forward),
            ],
            overlaps: vec!["4M".to_string(), "5M".to_string()],
        };

        match parse_path(path) {
            Err(err) => {
                panic!(format!("{:?}", err));
            }
            Ok((_res, p)) => assert_eq!(p, path_),
        }
    }

    #[test]
    fn can_parse_lines() {
        let input = "H	VN:Z:1.0
S	1	CAAATAAG
S	2	A
S	3	G
S	4	T
S	5	C
L	1	+	2	+	0M
L	1	+	3	+	0M
P	x	1+,3+,5+,6+,8+,9+,11+,12+,14+,15+	8M,1M,1M,3M,1M,19M,1M,4M,1M,11M";

        let lines = input.lines();
        let mut gfa = GFA::new();

        let gfa_correct = GFA {
            segments: vec![
                Segment::new("1", "CAAATAAG"),
                Segment::new("2", "A"),
                Segment::new("3", "G"),
                Segment::new("4", "T"),
                Segment::new("5", "C"),
            ],
            links: vec![
                Link::new("1", Orientation::Forward, "2", Orientation::Forward, "0M"),
                Link::new("1", Orientation::Forward, "3", Orientation::Forward, "0M"),
            ],
            paths: vec![Path::new(
                "x",
                vec![
                    "1+", "3+", "5+", "6+", "8+", "9+", "11+", "12+", "14+", "15+",
                ],
                vec!["8M", "1M", "1M", "3M", "1M", "19M", "1M", "4M", "1M", "11M"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            )],
            containments: vec![],
        };

        for l in lines {
            let p = parse_line(l);

            if let Ok((_, Line::Segment(s))) = p {
                gfa.segments.push(s);
            } else if let Ok((_, Line::Link(l))) = p {
                gfa.links.push(l);
            } else if let Ok((_, Line::Path(pt))) = p {
                gfa.paths.push(pt);
            }
        }

        assert_eq!(gfa_correct, gfa);
    }

    #[test]
    fn can_parse_gfa_file() {
        let gfa = parse_gfa(&PathBuf::from("./lil.gfa"));

        match gfa {
            None => panic!("Error parsing GFA file"),
            Some(g) => {
                let num_segs = g.segments.len();
                let num_links = g.links.len();
                let num_paths = g.paths.len();
                let num_conts = g.containments.len();

                assert_eq!(num_segs, 15);
                assert_eq!(num_links, 20);
                assert_eq!(num_conts, 0);
                assert_eq!(num_paths, 3);
            }
        }
    }

    #[test]
    fn can_stream_gfa_lines() {
        let file = File::open(&PathBuf::from("./lil.gfa")).unwrap();

        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut gfa_lines = parse_gfa_stream(&mut lines);

        assert_eq!(
            gfa_lines.next(),
            Some(Line::Header(Header {
                version: Some("1.0".to_string())
            }))
        );

        assert_eq!(
            gfa_lines.next(),
            Some(Line::Segment(Segment::new("1", "CAAATAAG")))
        );
        assert_eq!(
            gfa_lines.next(),
            Some(Line::Segment(Segment::new("2", "A")))
        );

        assert_eq!(
            gfa_lines.next(),
            Some(Line::Segment(Segment::new("3", "G")))
        );
    }
}
