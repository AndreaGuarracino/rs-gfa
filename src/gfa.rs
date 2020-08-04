use bstr::{BStr, BString, ByteSlice};
use serde::{Deserialize, Serialize};

use crate::optfields::*;

/// This module defines the various GFA line types, the GFA object,
/// and some utility functions and types.

/// Simple representation of a parsed GFA file, using a Vec<T> to
/// store each separate GFA line type.
#[derive(Default, Debug, Clone, PartialEq, PartialOrd)]
pub struct GFA<N, T: OptFields> {
    pub header: Header<T>,
    pub segments: Vec<Segment<N, T>>,
    pub links: Vec<Link<N, T>>,
    pub containments: Vec<Containment<N, T>>,
    pub paths: Vec<Path<T>>,
}

impl<N, T: OptFields> GFA<N, T> {
    /// Insert a GFA line (wrapped in the Line enum) into an existing
    /// GFA. Simply pushes it into the corresponding Vec in the GFA,
    /// or replaces the header, so there's no deduplication or sorting
    /// taking place.
    pub fn insert_line(&mut self, line: Line<N, T>) {
        use Line::*;
        match line {
            Header(h) => self.header = h,
            Segment(s) => self.segments.push(s),
            Link(s) => self.links.push(s),
            Containment(s) => self.containments.push(s),
            Path(s) => self.paths.push(s),
        }
    }
}

impl<T: OptFields> GFA<BString, T> {
    /// Creates a new GFA object whose segment names are all 64-bit
    /// unsigned integers. If there's any occurrence in the input GFA
    /// of a segment name that cannot be parsed into a usize, None is
    /// returned.
    pub fn usize_names(&self) -> Option<GFA<usize, T>> {
        let mut gfa = GFA::new();

        gfa.header = self.header.clone();

        for seg in self.segments.iter() {
            let new_seg = seg.usize_name()?;
            gfa.segments.push(new_seg);
        }

        for link in self.links.iter() {
            let new_link = link.usize_name()?;
            gfa.links.push(new_link);
        }

        for cont in self.containments.iter() {
            let new_cont = cont.usize_name()?;
            gfa.containments.push(new_cont);
        }

        for path in self.paths.iter() {
            if path.usize_segments() {
                gfa.paths.push(path.clone());
            } else {
                return None;
            }
        }

        Some(gfa)
    }
}

/// Consume a GFA object to produce an iterator over all the lines
/// contained within. The iterator first produces all segments, then
/// links, then containments, and finally paths.
pub fn gfa_into_iter<N, T: OptFields>(
    gfa: GFA<N, T>,
) -> impl Iterator<Item = Line<N, T>> {
    use Line::*;
    let segs = gfa.segments.into_iter().map(Segment);
    let links = gfa.links.into_iter().map(Link);
    let conts = gfa.containments.into_iter().map(Containment);
    let paths = gfa.paths.into_iter().map(Path);

    segs.chain(links).chain(conts).chain(paths)
}

impl<N: Default, T: OptFields> GFA<N, T> {
    pub fn new() -> Self {
        Default::default()
    }
}

/// Enum containing the different kinds of GFA lines.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum Line<N, T: OptFields> {
    Header(Header<T>),
    Segment(Segment<N, T>),
    Link(Link<N, T>),
    Containment(Containment<N, T>),
    Path(Path<T>),
}

/// The header line of a GFA graph
#[derive(Default, Debug, Clone, PartialEq, PartialOrd)]
pub struct Header<T: OptFields> {
    pub version: Option<BString>,
    pub optional: T,
}

/// A segment in a GFA graph. Generic over the name type, but
/// currently the parser is only defined for N = BString
#[derive(
    Default, Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize,
)]
pub struct Segment<N, T: OptFields> {
    pub name: N,
    pub sequence: BString,
    pub optional: T,
}

fn parse_usize(bs: &BString) -> Option<usize> {
    let s = std::str::from_utf8(bs.as_ref()).ok()?;
    s.parse::<usize>().ok()
}

impl<T: OptFields> Segment<BString, T> {
    pub fn new(name: &[u8], sequence: &[u8]) -> Self {
        Segment {
            name: BString::from(name),
            sequence: BString::from(sequence),
            optional: Default::default(),
        }
    }

    pub fn usize_name(&self) -> Option<Segment<usize, T>> {
        let ix = parse_usize(&self.name)?;
        Some(Segment {
            name: ix,
            sequence: self.sequence.clone(),
            optional: self.optional.clone(),
        })
    }
}

#[derive(
    Default, Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize,
)]
pub struct Link<N, T: OptFields> {
    pub from_segment: N,
    pub from_orient: Orientation,
    pub to_segment: N,
    pub to_orient: Orientation,
    pub overlap: BString,
    pub optional: T,
}

impl<T: OptFields> Link<BString, T> {
    pub fn new(
        from_segment: &[u8],
        from_orient: Orientation,
        to_segment: &[u8],
        to_orient: Orientation,
        overlap: &[u8],
    ) -> Link<BString, T> {
        Link {
            from_segment: from_segment.into(),
            from_orient,
            to_segment: to_segment.into(),
            to_orient,
            overlap: overlap.into(),
            optional: Default::default(),
        }
    }

    pub fn usize_name(&self) -> Option<Link<usize, T>> {
        let from_segment = parse_usize(&self.from_segment)?;
        let to_segment = parse_usize(&self.to_segment)?;
        Some(Link {
            from_segment,
            from_orient: self.from_orient,
            to_segment,
            to_orient: self.to_orient,
            overlap: self.overlap.clone(),
            optional: self.optional.clone(),
        })
    }
}

#[derive(
    Default, Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize,
)]
pub struct Containment<N, T: OptFields> {
    pub container_name: N,
    pub container_orient: Orientation,
    pub contained_name: N,
    pub contained_orient: Orientation,
    pub pos: usize,
    pub overlap: BString,
    pub optional: T,
}

impl<T: OptFields> Containment<BString, T> {
    pub fn usize_name(&self) -> Option<Containment<usize, T>> {
        let container_name = parse_usize(&self.container_name)?;
        let contained_name = parse_usize(&self.contained_name)?;
        Some(Containment {
            container_name,
            container_orient: self.container_orient,
            contained_name,
            contained_orient: self.contained_orient,
            pos: self.pos,
            overlap: self.overlap.clone(),
            optional: self.optional.clone(),
        })
    }
}

/// The step list that the path actually consists of is an unparsed
/// BString to keep memory down; use path.iter() to get an iterator
/// over the parsed path segments and orientations.
#[derive(
    Default, Debug, Clone, PartialEq, PartialOrd, Serialize, Deserialize,
)]
pub struct Path<T: OptFields> {
    pub path_name: BString,
    pub segment_names: BString,
    pub overlaps: Vec<BString>,
    pub optional: T,
}

impl<T: OptFields> Path<T> {
    /// `true` if each of the segment names of the path can be parsed to `usize`
    pub fn usize_segments(&self) -> bool {
        self.iter()
            .all(|(seg, _)| seg.iter().all(|b| b.is_ascii_digit()))
    }
}

/*
pub enum PathSegments {
    Unparsed(BString),
    Parsed(Vec<(BString, Orientation)>),
}

impl PathSegments {
    /// Produces an iterator over the parsed segments of the given path
    pub fn iter<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = (&'a BStr, Orientation)> + 'a> {
        match self {
            PathSegments::Unparsed(bs) => {
                let iter: _ = bs.split_str(b",").map(parse_path_segment);
                Box::new(iter)
            }
            PathSegments::Parsed(steps) => {
                let iter: _ = steps.iter().map(|(b, o)| (b.as_ref(), *o));
                Box::new(iter)
            }
        }
    }
}
*/

/// Parses a segment in a Path's segment_names into a segment name and
/// orientation
fn parse_path_segment(input: &[u8]) -> (&'_ BStr, Orientation) {
    use Orientation::*;
    let last = input.len() - 1;
    let orient = match input[last] {
        b'+' => Forward,
        b'-' => Backward,
        _ => panic!("Path segment did not include orientation"),
    };
    let seg = &input[..last];
    (seg.as_ref(), orient)
}

impl<T: OptFields> Path<T> {
    /// Produces an iterator over the parsed segments of the given path
    pub fn iter(&self) -> impl Iterator<Item = (&'_ BStr, Orientation)> {
        self.segment_names.split_str(b",").map(parse_path_segment)
    }
}

/// Represents segment orientation/strand
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum Orientation {
    Forward,
    Backward,
}

// It makes sense for forward to be the default
impl Default for Orientation {
    fn default() -> Orientation {
        Orientation::Forward
    }
}

impl From<Orientation> for bool {
    fn from(o: Orientation) -> bool {
        match o {
            Orientation::Forward => true,
            Orientation::Backward => false,
        }
    }
}

impl Orientation {
    pub fn is_reverse(&self) -> bool {
        !bool::from(*self)
    }

    pub fn from_bytes<T: AsRef<[u8]>>(bs: T) -> Option<Self> {
        match bs.as_ref() {
            b"+" => Some(Orientation::Forward),
            b"-" => Some(Orientation::Backward),
            _ => None,
        }
    }
}

impl std::str::FromStr for Orientation {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "+" => Ok(Self::Forward),
            "-" => Ok(Self::Backward),
            _ => Err("Could not parse orientation (was not + or -)"),
        }
    }
}

impl std::fmt::Display for Orientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let sym = match self {
            Self::Forward => '+',
            Self::Backward => '-',
        };
        write!(f, "{}", sym)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_iter() {
        use Orientation::*;
        let path = Path {
            path_name: "14".into(),
            segment_names: "11+,12-,13+".into(),
            overlaps: vec!["4M".into(), "5M".into()],
            optional: (),
        };
        let mut path_iter = path.iter();
        assert_eq!(Some(("11".into(), Forward)), path_iter.next());
        assert_eq!(Some(("12".into(), Backward)), path_iter.next());
        assert_eq!(Some(("13".into(), Forward)), path_iter.next());
        assert_eq!(None, path_iter.next());
    }
}
