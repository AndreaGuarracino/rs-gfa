use crate::{gfa::Line, parser::GFAParser};

use anyhow::{bail, Result};

use memmap::Mmap;

use std::fs::File;
use std::io::prelude::*;

use bstr::ByteSlice;

#[derive(Debug)]
pub struct MmapGFA {
    pub cursor: std::io::Cursor<Mmap>,
    pub line_buf: Vec<u8>,
    pub current_line_len: usize,
    pub last_buf_offset: usize,
    pub parser: GFAParser<usize, ()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineType {
    Segment,
    Link,
    Path,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineIndices {
    pub segments: Vec<(usize, usize)>,
    pub links: Vec<usize>,
    pub paths: Vec<usize>,
    /// byte length of each path line, in the same order as `paths`.
    /// Only `build_index_par` fills this; the serial scan leaves it
    /// empty.
    pub path_lengths: Vec<usize>,
}

impl MmapGFA {
    pub fn new(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let cursor = std::io::Cursor::new(mmap);
        let line_buf = Vec::with_capacity(1024);
        let current_line_len = 0;
        let last_buf_offset = 0;

        let parser = GFAParser::new();

        Ok(Self {
            cursor,
            line_buf,
            current_line_len,
            last_buf_offset,
            parser,
        })
    }

    pub fn reset_position(&mut self) -> u64 {
        let cur_pos = self.cursor.position();
        self.cursor.set_position(0);
        cur_pos
    }

    pub fn set_position(&mut self, new_pos: u64) -> u64 {
        let cur_pos = self.cursor.position();
        self.cursor.set_position(new_pos);
        cur_pos
    }
    pub fn get_ref(&self) -> &[u8] {
        self.cursor.get_ref().as_ref()
    }

    pub fn get_parser(&self) -> &GFAParser<usize, ()> {
        &self.parser
    }

    pub fn next_line(&mut self) -> Result<&[u8]> {
        self.line_buf.clear();

        self.last_buf_offset = self.cursor.position() as usize;

        let n_read = self.cursor.read_until(b'\n', &mut self.line_buf)?;

        self.current_line_len = n_read;

        Ok(&self.line_buf[..n_read])
    }

    pub fn read_line_at(&mut self, offset: usize) -> Result<&[u8]> {
        self.cursor.set_position(offset as u64);
        self.next_line()
    }

    /// Ask the kernel to populate the page tables for the whole
    /// mapping up front, in parallel.
    ///
    /// Parsing a multi-gigabyte GFA otherwise takes a minor fault per
    /// page, one at a time, interleaved with the parsing work. Doing
    /// it in bulk beforehand is the same pages but far fewer traps.
    ///
    /// Best effort: if the kernel does not support populating, this
    /// falls back to a readahead hint, and if that fails too it does
    /// nothing.
    #[cfg(target_os = "linux")]
    pub fn prefault(&self) {
        use rayon::prelude::*;

        // Linux 5.14 and later
        const MADV_POPULATE_READ: libc::c_int = 22;

        let data: &[u8] = self.cursor.get_ref();
        let len = data.len();

        if len == 0 {
            return;
        }

        let base = data.as_ptr() as usize;
        let chunk = 64usize << 20;

        // Populating more than fits just evicts the early pages before
        // the parse reaches them, so fall back to a hint instead.
        // MemAvailable rather than MemFree, since reclaimable page
        // cache counts as available.
        let available = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|meminfo| {
                meminfo
                    .lines()
                    .find(|l| l.starts_with("MemAvailable:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|kb| kb.parse::<usize>().ok())
            })
            .map(|kb| kb.saturating_mul(1024));

        // half, because the caller builds its own structures from this
        if available.map(|avail| len > avail / 2).unwrap_or(false) {
            unsafe {
                libc::madvise(
                    base as *mut libc::c_void,
                    len,
                    libc::MADV_WILLNEED,
                );
            }
            return;
        }

        let offsets: Vec<usize> = (0..len).step_by(chunk).collect();

        let populated: usize = offsets
            .par_iter()
            .map(|&off| {
                let this = chunk.min(len - off);

                let res = unsafe {
                    libc::madvise(
                        (base + off) as *mut libc::c_void,
                        this,
                        MADV_POPULATE_READ,
                    )
                };

                if res == 0 {
                    1
                } else {
                    0
                }
            })
            .sum();

        if populated == 0 {
            unsafe {
                libc::madvise(
                    base as *mut libc::c_void,
                    len,
                    libc::MADV_WILLNEED,
                );
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn prefault(&self) {}

    /// As `build_index`, but scans the mapping in parallel.
    ///
    /// The file is split into one chunk per thread, each split point
    /// moved forward to the next line break so no line is cut, and the
    /// per-chunk results are concatenated in file order. That keeps the
    /// segment, link and path orders identical to the serial scan.
    pub fn build_index_par(&self) -> Result<LineIndices> {
        use rayon::prelude::*;

        let data: &[u8] = self.cursor.get_ref();
        let len = data.len();

        if len == 0 {
            return Ok(LineIndices {
                segments: Vec::new(),
                links: Vec::new(),
                paths: Vec::new(),
                path_lengths: Vec::new(),
            });
        }

        let threads = rayon::current_num_threads().max(1);
        let target = (len / threads).max(1 << 20);

        // split points, each advanced to just past a line break
        let mut bounds: Vec<usize> = Vec::with_capacity(threads + 1);
        bounds.push(0);

        let mut at = target;
        while at < len {
            let cut = match memchr::memchr(b'\n', &data[at..]) {
                Some(ix) => at + ix + 1,
                None => len,
            };

            if cut >= len {
                break;
            }

            if cut > *bounds.last().unwrap() {
                bounds.push(cut);
            }

            at = cut + target;
        }
        bounds.push(len);

        let chunks: Vec<(usize, usize)> =
            bounds.windows(2).map(|w| (w[0], w[1])).collect();

        let per_chunk: Vec<LineIndices> = chunks
            .par_iter()
            .map(|&(start, end)| {
                let mut segments = Vec::new();
                let mut links = Vec::new();
                let mut paths = Vec::new();
                let mut path_lengths = Vec::new();

                let mut line_start = start;

                for nl in memchr::memchr_iter(b'\n', &data[start..end]) {
                    let line_end = start + nl + 1;
                    let length = line_end - line_start;

                    match data[line_start] {
                        b'S' => segments.push((line_start, length)),
                        b'L' => links.push(line_start),
                        b'P' => {
                            paths.push(line_start);
                            path_lengths.push(length);
                        }
                        _ => (),
                    }

                    line_start = line_end;
                }

                // a final line with no trailing break
                if line_start < end {
                    let length = end - line_start;

                    match data[line_start] {
                        b'S' => segments.push((line_start, length)),
                        b'L' => links.push(line_start),
                        b'P' => {
                            paths.push(line_start);
                            path_lengths.push(length);
                        }
                        _ => (),
                    }
                }

                LineIndices {
                    segments,
                    links,
                    paths,
                    path_lengths,
                }
            })
            .collect();

        let mut segments = Vec::new();
        let mut links = Vec::new();
        let mut paths = Vec::new();
        let mut path_lengths = Vec::new();

        for chunk in per_chunk {
            segments.extend(chunk.segments);
            links.extend(chunk.links);
            paths.extend(chunk.paths);
            path_lengths.extend(chunk.path_lengths);
        }

        Ok(LineIndices {
            segments,
            links,
            paths,
            path_lengths,
        })
    }

    pub fn build_index(&mut self) -> Result<LineIndices> {
        let start_position = self.cursor.position();
        let current_line_len = self.current_line_len;
        let last_buf_offset = self.last_buf_offset;

        let mut segments = Vec::new();
        let mut links = Vec::new();
        let mut paths = Vec::new();

        self.cursor.set_position(0);

        let mut line_start = 0;

        loop {
            let line = self.next_line()?;
            let length = line.len();

            if let Some(ref byte) = line.first() {
                match byte {
                    b'S' => {
                        segments.push((line_start, length));
                    }
                    b'L' => {
                        links.push(line_start);
                    }
                    b'P' => {
                        paths.push(line_start);
                    }
                    _ => (),
                };

                line_start += line.len();
            } else {
                break;
            }
        }

        self.cursor.set_position(start_position);
        self.current_line_len = current_line_len;
        self.last_buf_offset = last_buf_offset;

        let res = LineIndices {
            segments,
            links,
            paths,
            path_lengths: Vec::new(),
        };

        Ok(res)
    }

    pub fn current_line(&self) -> &[u8] {
        &self.line_buf[..self.current_line_len]
    }

    pub fn current_line_name(&self) -> Option<&[u8]> {
        let mut iter = self.line_buf.split_str("\t");
        let _lt = iter.next()?;
        let name = iter.next()?;
        Some(name)
    }

    pub fn parse_current_line(&self) -> Result<Line<usize, ()>> {
        let line = self.current_line();
        if line.is_empty() {
            bail!("Line at offset {} is empty", self.last_buf_offset);
        }

        let gfa_line = self.parser.parse_gfa_line(line)?;
        Ok(gfa_line)
    }



}
